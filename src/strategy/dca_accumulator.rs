//! DCA Accumulator Strategy — a conservative, long-term, no-leverage strategy.
//!
//! Core philosophy: **Time in the market beats timing the market.**
//!
//! Mechanism:
//! - Dollar-Cost Average into BTC at regular intervals (cooldown-gated)
//! - RSI filter: skip buys when overbought (RSI > 70), buy extra when oversold (< 30)
//! - RSI profit exit: sell the profit portion when RSI > 80 (exceedingly overbought)
//! - SMA trend filter: only buy when price is above 200-tick SMA (long-term uptrend)
//! - Volatility scaling: reduce size in high-vol, halt in extreme-vol
//! - Trailing stop-loss: protective sell if price drops 12% from peak
//! - Max 50% of portfolio in crypto at any time
//! - No shorts, no leverage — ever

use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::types::*;
use rust_decimal::prelude::ToPrimitive;

use super::indicators::{self, PriceBuffer};
use super::risk::{RiskManager, RiskParams};
use super::traits::ExecutionClient;

/// Configuration specific to the DCA accumulator.
#[derive(Debug, Clone)]
pub struct DcaConfig {
    /// RSI period (default: 14)
    pub rsi_period: usize,
    /// SMA trend filter period (default: 200)
    pub sma_period: usize,
    /// Volatility lookback window (default: 20)
    pub vol_window: usize,
    /// RSI threshold above which we skip buys
    pub rsi_overbought: Decimal,
    /// RSI threshold below which we increase buy size
    pub rsi_oversold: Decimal,
    /// RSI threshold above which we sell the profit portion of the position
    pub rsi_exit_threshold: Decimal,
    /// Multiplier applied to base notional when RSI is oversold
    pub oversold_multiplier: Decimal,
}

impl Default for DcaConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            sma_period: 200,
            vol_window: 20,
            rsi_overbought: dec!(70),
            rsi_oversold: dec!(30),
            rsi_exit_threshold: dec!(80),
            oversold_multiplier: dec!(1.5),
        }
    }
}

/// Run the DCA accumulator strategy as a long-lived tokio task.
pub async fn run_dca_strategy(
    client: impl ExecutionClient,
    mut tick_rx: mpsc::Receiver<MarketTick>,
    mut report_rx: mpsc::Receiver<ExecutionReport>,
    initial_balance: Decimal,
    snapshot_tx: Option<mpsc::Sender<PortfolioSnapshot>>,
    mut command_rx: mpsc::Receiver<DashboardCommand>,
) {
    let dca_cfg = DcaConfig::default();
    let risk_params = RiskParams::default();
    let mut risk = RiskManager::new(risk_params, initial_balance);

    // We need enough history for the longest indicator (SMA-200)
    let buf_cap = dca_cfg.sma_period + 1;
    let mut price_buf = PriceBuffer::new(buf_cap);

    let mut order_id: u64 = 0;
    let mut quote_balance = initial_balance;
    let mut crypto_qty = dec!(0);
    let mut total_cost_basis = dec!(0);
    let mut tick_count: u64 = 0;
    let mut event_history: VecDeque<String> = VecDeque::with_capacity(50);
    let mut last_rsi: Option<f64> = None;
    let mut last_price = dec!(0); // track for manual commands
    let mut last_symbol = String::from("BTCUSDT");

    macro_rules! push_event {
        ($msg:expr) => {
            let msg = format!("[{}] {}", Utc::now().format("%H:%M:%S"), $msg);
            info!("{}", msg);
            event_history.push_front(msg);
            if event_history.len() > 50 {
                event_history.pop_back();
            }
            emit_snapshot!(Utc::now().timestamp_millis() as u64);
        };
    }

    macro_rules! emit_snapshot {
        ($timestamp:expr) => {
            if let Some(ref tx) = snapshot_tx {
                let crypto_value = crypto_qty * last_price;
                let portfolio_value = quote_balance + crypto_value;
                let alloc_pct = if portfolio_value > dec!(0) {
                    crypto_value / portfolio_value * dec!(100)
                } else { dec!(0) };
                let unrealized_pnl = if crypto_qty > dec!(0) {
                    crypto_value - total_cost_basis
                } else { dec!(0) };
                let _ = tx.try_send(PortfolioSnapshot {
                    timestamp: $timestamp,
                    price: last_price,
                    quote_balance,
                    crypto_qty,
                    portfolio_value,
                    allocation_pct: alloc_pct,
                    cost_basis: total_cost_basis,
                    unrealized_pnl,
                    rsi: last_rsi,
                    event_history: event_history.iter().cloned().collect(),
                });
            }
        };
    }

    // Subsample ticks — we don't need to evaluate on every aggTrade.
    // Process every Nth tick to build meaningful candle-like data points.
    let tick_sample_rate: u64 = 100;

    info!(
        rsi_period = dca_cfg.rsi_period,
        sma_period = dca_cfg.sma_period,
        max_alloc = %risk.params.max_allocation,
        trailing_stop = %risk.params.trailing_stop_pct,
        rsi_exit = %dca_cfg.rsi_exit_threshold,
        "DCA Accumulator strategy started"
    );

    loop {
        tokio::select! {
            Some(tick) = tick_rx.recv() => {
                tick_count += 1;
                last_price = tick.price;
                last_symbol = tick.symbol.clone();

                // Subsample: only process every Nth tick
                if tick_count % tick_sample_rate != 0 {
                    if tick_count % (tick_sample_rate / 2).max(1) == 0 {
                        emit_snapshot!(tick.timestamp);
                    }
                    continue;
                }

                let price = tick.price;
                price_buf.push(price);

                // Update trailing stop tracker
                if crypto_qty > dec!(0) {
                    risk.update_price_peak(price);
                }

                // Update portfolio value for circuit breaker
                let crypto_value = crypto_qty * price;
                let portfolio_value = quote_balance + crypto_value;
                risk.update_portfolio_value(portfolio_value);

                // ── Dashboard Snapshot ─────────────────────────────
                emit_snapshot!(tick.timestamp);

                // ── Trailing Stop Check ──────────────────────────
                if crypto_qty > dec!(0) && risk.trailing_stop_triggered(price) {
                    order_id += 1;
                    let sell_order = OrderRequest {
                        id: order_id,
                        symbol: tick.symbol.clone(),
                        side: Side::Sell,
                        qty: crypto_qty,
                        price,
                    };
                    warn!(
                        id = order_id,
                        price = %price,
                        qty = %crypto_qty,
                        "TRAILING STOP triggered — selling entire position"
                    );
                    if let Err(e) = client.submit_order(sell_order).await {
                        warn!(error = %e, "Failed to submit stop-loss sell");
                    }
                    risk.record_trade(tick.timestamp);
                    push_event!(format!("TRAILING STOP — sold {} BTC @ ${}", crypto_qty, price));
                    continue;
                }

                // ── Need minimum data for indicators ─────────────
                if !price_buf.is_full() {
                    if tick_count % (tick_sample_rate * 50) == 0 {
                        info!(
                            buffered = price_buf.len(),
                            needed = buf_cap,
                            "Warming up indicator buffer…"
                        );
                    }
                    continue;
                }

                // ── Compute Indicators ───────────────────────────
                let prices = price_buf.prices();
                let current_rsi = indicators::rsi(prices, dca_cfg.rsi_period);
                let sma_200 = indicators::sma(prices, dca_cfg.sma_period);
                let vol = indicators::volatility_mad(prices, dca_cfg.vol_window);

                // Track RSI for dashboard
                if let Some(r) = current_rsi {
                    last_rsi = r.to_f64();
                }

                // ── RSI Profit-Taking Exit ────────────────────────
                // When RSI is exceedingly overbought, sell only the
                // profit portion — keep the cost basis invested.
                if let Some(rsi_val) = current_rsi {
                    if rsi_val > dca_cfg.rsi_exit_threshold
                        && crypto_qty > dec!(0)
                        && total_cost_basis > dec!(0)
                    {
                        let avg_entry = total_cost_basis / crypto_qty;
                        if price > avg_entry {
                            // Profit portion: how many coins represent
                            // the unrealized gain at current price
                            // cost_basis_qty = total_cost_basis / price
                            // profit_qty     = crypto_qty - cost_basis_qty
                            let cost_basis_qty = total_cost_basis / price;
                            let profit_qty = crypto_qty - cost_basis_qty;

                            if profit_qty > dec!(0) {
                                order_id += 1;
                                let sell_order = OrderRequest {
                                    id: order_id,
                                    symbol: tick.symbol.clone(),
                                    side: Side::Sell,
                                    qty: profit_qty,
                                    price,
                                };
                                info!(
                                    id = order_id,
                                    rsi = %rsi_val,
                                    price = %price,
                                    avg_entry = %avg_entry,
                                    profit_qty = %profit_qty,
                                    kept_qty = %cost_basis_qty,
                                    "RSI EXIT — selling profit portion"
                                );
                                if let Err(e) = client.submit_order(sell_order).await {
                                    warn!(error = %e, "Failed to submit RSI exit sell");
                                }
                                risk.record_trade(tick.timestamp);
                                push_event!(format!("RSI EXIT — sold {} BTC profit @ ${}", profit_qty, price));
                                continue;
                            }
                        }
                    }
                }

                // ── SMA Trend Filter ─────────────────────────────
                // Only buy when price is above the long-term SMA
                // (confirms we're in an uptrend)
                if let Some(sma) = sma_200 {
                    if price < sma {
                        continue; // Below trend — sit on hands
                    }
                }

                // ── RSI Filter ───────────────────────────────────
                if let Some(rsi_val) = current_rsi {
                    if rsi_val > dca_cfg.rsi_overbought {
                        continue; // Overbought — skip this DCA window
                    }
                }

                // ── Risk Manager Approval ────────────────────────
                let approved = risk.approve_buy(
                    tick.timestamp,
                    quote_balance,
                    crypto_value,
                    vol,
                );

                let mut notional = match approved {
                    Some(n) => n,
                    None => continue,
                };

                // ── RSI Oversold Bonus ───────────────────────────
                if let Some(rsi_val) = current_rsi {
                    if rsi_val < dca_cfg.rsi_oversold {
                        notional = (notional * dca_cfg.oversold_multiplier)
                            .min(risk.params.max_trade_notional)
                            .min(quote_balance);
                        info!(rsi = %rsi_val, boosted = %notional, "RSI oversold — boosting buy");
                    }
                }

                // ── Submit Buy Order ─────────────────────────────
                if price > dec!(0) {
                    let qty = notional / price;
                    if qty > dec!(0) {
                        order_id += 1;
                        let order = OrderRequest {
                            id: order_id,
                            symbol: tick.symbol.clone(),
                            side: Side::Buy,
                            qty,
                            price,
                        };
                        info!(
                            id = order_id,
                            notional = %notional,
                            qty = %qty,
                            price = %price,
                            rsi = ?current_rsi,
                            vol = ?vol,
                            alloc_pct = %(crypto_value / portfolio_value * dec!(100)),
                            "DCA buy signal"
                        );
                        if let Err(e) = client.submit_order(order).await {
                            warn!(error = %e, "Failed to submit DCA buy");
                        }
                        risk.record_trade(tick.timestamp);
                        push_event!(format!("DCA BUY — {} BTC @ ${} (${} notional)", qty, price, notional));
                    }
                }
            }

            Some(report) = report_rx.recv() => {
                match report.status {
                    FillStatus::Filled => {
                        let notional = report.fill_price * report.filled_qty;
                        match report.side {
                            Side::Buy => {
                                quote_balance -= notional + report.fee;
                                crypto_qty += report.filled_qty;
                                total_cost_basis += notional + report.fee;
                                info!(
                                    order_id = report.order_id,
                                    side = "BUY",
                                    fill_price = %report.fill_price,
                                    qty = %report.filled_qty,
                                    fee = %report.fee,
                                    quote_bal = %quote_balance,
                                    crypto_qty = %crypto_qty,
                                    avg_entry = %(total_cost_basis / crypto_qty),
                                    "Fill confirmed"
                                );
                            }
                            Side::Sell => {
                                quote_balance += notional - report.fee;
                                crypto_qty -= report.filled_qty;
                                // Scale down cost basis proportionally
                                if crypto_qty <= dec!(0) {
                                    crypto_qty = dec!(0);
                                    total_cost_basis = dec!(0);
                                    risk.reset_trailing_stop();
                                } else {
                                    // Keep cost basis proportional to remaining position
                                    let sold_fraction = report.filled_qty / (crypto_qty + report.filled_qty);
                                    total_cost_basis -= total_cost_basis * sold_fraction;
                                }
                                info!(
                                    order_id = report.order_id,
                                    side = "SELL",
                                    fill_price = %report.fill_price,
                                    qty = %report.filled_qty,
                                    fee = %report.fee,
                                    quote_bal = %quote_balance,
                                    crypto_qty = %crypto_qty,
                                    cost_basis = %total_cost_basis,
                                    "Fill confirmed — position reduced"
                                );
                            }
                        }
                    }
                    FillStatus::Rejected => {
                        warn!(order_id = report.order_id, "Order rejected by engine");
                    }
                    FillStatus::PartialFill => {
                        info!(
                            order_id = report.order_id,
                            filled = %report.filled_qty,
                            "Partial fill"
                        );
                    }
                }
            }

            Some(cmd) = command_rx.recv() => {
                match cmd {
                    DashboardCommand::ManualBuy { notional } => {
                        if last_price <= dec!(0) {
                            warn!("Manual buy ignored — no price data yet");
                            continue;
                        }
                        if notional > quote_balance {
                            push_event!(format!("REJECTED — insufficient cash (${} available)", quote_balance));
                            continue;
                        }
                        let qty = notional / last_price;
                        if qty > dec!(0) {
                            order_id += 1;
                            let order = OrderRequest {
                                id: order_id,
                                symbol: last_symbol.clone(),
                                side: Side::Buy,
                                qty,
                                price: last_price,
                            };
                            info!(
                                id = order_id,
                                notional = %notional,
                                qty = %qty,
                                price = %last_price,
                                "MANUAL BUY submitted"
                            );
                            if let Err(e) = client.submit_order(order).await {
                                warn!(error = %e, "Failed to submit manual buy");
                            }
                            push_event!(format!("MANUAL BUY — {} BTC @ ${} (${} notional)", qty, last_price, notional));
                        }
                    }
                    DashboardCommand::PanicSell => {
                        if crypto_qty <= dec!(0) {
                            warn!("Panic sell ignored — no position");
                            push_event!("PANIC SELL — no position to sell".to_string());
                            continue;
                        }
                        if last_price <= dec!(0) {
                            warn!("Panic sell ignored — no price data yet");
                            continue;
                        }
                        order_id += 1;
                        let order = OrderRequest {
                            id: order_id,
                            symbol: last_symbol.clone(),
                            side: Side::Sell,
                            qty: crypto_qty,
                            price: last_price,
                        };
                        if let Err(e) = client.submit_order(order).await {
                            warn!(error = %e, "Failed to submit panic sell");
                        }
                        push_event!(format!("🚨 PANIC SELL — {} BTC @ ${}", crypto_qty, last_price));
                    }
                }
            }

            else => {
                info!("All channels closed — DCA strategy shutting down.");
                break;
            }
        }
    }
}
