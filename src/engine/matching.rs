//! Paper matching engine — simulates order fills with slippage and fees.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::Config;
use crate::types::*;

use super::account::VirtualAccount;

/// Run the paper engine as a long-lived tokio task.
///
/// It consumes [`OrderRequest`]s from the strategy, simulates fills against
/// the last known market price, and emits [`ExecutionReport`]s back.
pub async fn run_engine(
    config: Config,
    mut order_rx: mpsc::Receiver<OrderRequest>,
    report_tx: mpsc::Sender<ExecutionReport>,
    mut event_rx: mpsc::Receiver<MarketEvent>,
) {
    let mut account = VirtualAccount::new(config.initial_balance);
    let mut last_price: Option<Decimal> = None;

    info!("Paper engine started.");

    loop {
        tokio::select! {
            // Continuously update the latest market price
            Some(event) = event_rx.recv() => {
                if let MarketEvent::Tick(tick) = event {
                    last_price = Some(tick.price);
                }
            }

            // Process incoming order requests
            Some(order) = order_rx.recv() => {
                let report = match last_price {
                    Some(market_price) => {
                        simulate_fill(&config, &mut account, &order, market_price)
                    }
                    None => {
                        warn!(order_id = order.id, "No market price yet — rejecting order.");
                        ExecutionReport {
                            order_id: order.id,
                            side: order.side,
                            status: FillStatus::Rejected,
                            filled_qty: dec!(0),
                            fill_price: dec!(0),
                            fee: dec!(0),
                        }
                    }
                };

                info!(
                    order_id = report.order_id,
                    status = ?report.status,
                    fill_price = %report.fill_price,
                    qty = %report.filled_qty,
                    fee = %report.fee,
                    balance = %account.quote_balance,
                    "Execution report"
                );

                if report_tx.send(report).await.is_err() {
                    warn!("Report channel closed — shutting down engine.");
                    return;
                }
            }

            // Both channels closed → exit cleanly
            else => {
                info!("All channels closed — engine shutting down.");
                break;
            }
        }
    }
}

/// Simulate a single order fill with slippage and fees.
pub(crate) fn simulate_fill(
    config: &Config,
    account: &mut VirtualAccount,
    order: &OrderRequest,
    market_price: Decimal,
) -> ExecutionReport {
    let base = VirtualAccount::base_asset(&order.symbol);

    // Apply simulated slippage
    let slippage_adjusted = match order.side {
        Side::Buy => market_price * (dec!(1) + config.slippage_bps),
        Side::Sell => market_price * (dec!(1) - config.slippage_bps),
    };

    let notional = slippage_adjusted * order.qty;
    let fee = notional * config.fee_rate;
    let total_cost = notional + fee;

    match order.side {
        Side::Buy => {
            if account.quote_balance < total_cost {
                return ExecutionReport {
                    order_id: order.id,
                    side: order.side,
                    status: FillStatus::Rejected,
                    filled_qty: dec!(0),
                    fill_price: slippage_adjusted,
                    fee: dec!(0),
                };
            }
            account.quote_balance -= total_cost;
            account.credit_base(base, order.qty);
            account.total_fees_paid += fee;
        }
        Side::Sell => {
            if account.holding(base) < order.qty {
                return ExecutionReport {
                    order_id: order.id,
                    side: order.side,
                    status: FillStatus::Rejected,
                    filled_qty: dec!(0),
                    fill_price: slippage_adjusted,
                    fee: dec!(0),
                };
            }
            account.debit_base(base, order.qty);
            account.quote_balance += notional - fee;
            account.total_fees_paid += fee;
        }
    }

    ExecutionReport {
        order_id: order.id,
        side: order.side,
        status: FillStatus::Filled,
        filled_qty: order.qty,
        fill_price: slippage_adjusted,
        fee,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::account::VirtualAccount;

    fn test_config() -> Config {
        Config {
            ws_url: String::new(),
            symbol: "btcusdt".into(),
            initial_balance: dec!(10_000),
            fee_rate: dec!(0.001),       // 10 bps
            slippage_bps: dec!(0.0005),  // 5 bps
            channel_buffer: 16,
        }
    }

    fn buy_order(id: u64, qty: Decimal, price: Decimal) -> OrderRequest {
        OrderRequest {
            id,
            symbol: "BTCUSDT".into(),
            side: Side::Buy,
            qty,
            price,
        }
    }

    fn sell_order(id: u64, qty: Decimal, price: Decimal) -> OrderRequest {
        OrderRequest {
            id,
            symbol: "BTCUSDT".into(),
            side: Side::Sell,
            qty,
            price,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Buy Fills
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn buy_fill_deducts_balance_and_credits_holding() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(10_000));
        let order = buy_order(1, dec!(0.1), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        assert_eq!(report.status, FillStatus::Filled);
        assert_eq!(report.side, Side::Buy);
        assert_eq!(report.filled_qty, dec!(0.1));
        assert!(acct.quote_balance < dec!(10_000), "Balance should decrease after buy");
        assert_eq!(acct.holding("BTC"), dec!(0.1));
    }

    #[test]
    fn buy_slippage_increases_fill_price() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));
        let order = buy_order(1, dec!(1.0), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        // Buy slippage: 50000 * (1 + 0.0005) = 50025
        assert_eq!(report.fill_price, dec!(50025.0000));
        assert!(report.fill_price > dec!(50_000), "Buy slippage should increase price");
    }

    #[test]
    fn buy_fee_correctly_calculated() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));
        let order = buy_order(1, dec!(1.0), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        // Notional = 50025 * 1.0 = 50025
        // Fee = 50025 * 0.001 = 50.025
        assert_eq!(report.fee, dec!(50.0250000));
        assert_eq!(acct.total_fees_paid, dec!(50.0250000));
    }

    #[test]
    fn buy_total_cost_is_notional_plus_fee() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));
        let order = buy_order(1, dec!(1.0), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        let expected_cost = report.fill_price * report.filled_qty + report.fee;
        let actual_deducted = dec!(100_000) - acct.quote_balance;
        assert_eq!(actual_deducted, expected_cost);
    }

    #[test]
    fn buy_rejected_insufficient_balance() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100)); // only $100
        let order = buy_order(1, dec!(1.0), dec!(50_000)); // needs ~$50k

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        assert_eq!(report.status, FillStatus::Rejected);
        assert_eq!(report.filled_qty, dec!(0));
        assert_eq!(report.fee, dec!(0));
        assert_eq!(acct.quote_balance, dec!(100)); // unchanged
        assert_eq!(acct.holding("BTC"), dec!(0));   // no credit
    }

    // ═══════════════════════════════════════════════════════════════════
    // Sell Fills
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn sell_fill_credits_balance_and_debits_holding() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(0));
        acct.credit_base("BTC", dec!(1.0));

        let order = sell_order(1, dec!(0.5), dec!(60_000));
        let report = simulate_fill(&cfg, &mut acct, &order, dec!(60_000));

        assert_eq!(report.status, FillStatus::Filled);
        assert_eq!(report.side, Side::Sell);
        assert!(acct.quote_balance > dec!(0), "Balance should increase after sell");
        assert_eq!(acct.holding("BTC"), dec!(0.5));
    }

    #[test]
    fn sell_slippage_decreases_fill_price() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(0));
        acct.credit_base("BTC", dec!(1.0));

        let order = sell_order(1, dec!(1.0), dec!(50_000));
        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        // Sell slippage: 50000 * (1 - 0.0005) = 49975
        assert_eq!(report.fill_price, dec!(49975.0000));
        assert!(report.fill_price < dec!(50_000), "Sell slippage should decrease price");
    }

    #[test]
    fn sell_proceeds_net_of_fee() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(0));
        acct.credit_base("BTC", dec!(1.0));

        let order = sell_order(1, dec!(1.0), dec!(50_000));
        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        let expected_proceeds = report.fill_price * report.filled_qty - report.fee;
        assert_eq!(acct.quote_balance, expected_proceeds);
    }

    #[test]
    fn sell_rejected_insufficient_holdings() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(10_000));
        // No BTC held
        let order = sell_order(1, dec!(1.0), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));

        assert_eq!(report.status, FillStatus::Rejected);
        assert_eq!(report.filled_qty, dec!(0));
        assert_eq!(acct.quote_balance, dec!(10_000)); // unchanged
    }

    #[test]
    fn sell_rejected_partial_insufficient() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(10_000));
        acct.credit_base("BTC", dec!(0.5)); // only 0.5 BTC

        let order = sell_order(1, dec!(1.0), dec!(50_000)); // trying to sell 1.0

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));
        assert_eq!(report.status, FillStatus::Rejected);
        assert_eq!(acct.holding("BTC"), dec!(0.5)); // unchanged
    }

    // ═══════════════════════════════════════════════════════════════════
    // Determinism & Conservation
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn buy_then_sell_roundtrip_loses_to_fees_and_slippage() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));

        // Buy 1 BTC at 50k
        let buy = buy_order(1, dec!(1.0), dec!(50_000));
        let buy_report = simulate_fill(&cfg, &mut acct, &buy, dec!(50_000));
        assert_eq!(buy_report.status, FillStatus::Filled);

        // Sell 1 BTC at same market price
        let sell = sell_order(2, dec!(1.0), dec!(50_000));
        let sell_report = simulate_fill(&cfg, &mut acct, &sell, dec!(50_000));
        assert_eq!(sell_report.status, FillStatus::Filled);

        // Should have LESS than starting balance (lost to fees + slippage)
        assert!(acct.quote_balance < dec!(100_000),
            "Roundtrip should lose money to fees/slippage, got {}", acct.quote_balance);
        assert_eq!(acct.holding("BTC"), dec!(0));
    }

    #[test]
    fn multiple_buys_accumulate_holdings() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));

        for i in 1..=5 {
            let order = buy_order(i, dec!(0.1), dec!(50_000));
            let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));
            assert_eq!(report.status, FillStatus::Filled);
        }

        assert_eq!(acct.holding("BTC"), dec!(0.5));
        assert!(acct.quote_balance < dec!(100_000));
        assert!(acct.total_fees_paid > dec!(0));
    }

    #[test]
    fn fees_accumulate_across_trades() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(100_000));

        let buy = buy_order(1, dec!(1.0), dec!(50_000));
        let buy_report = simulate_fill(&cfg, &mut acct, &buy, dec!(50_000));
        let fee1 = buy_report.fee;

        let sell = sell_order(2, dec!(1.0), dec!(50_000));
        let sell_report = simulate_fill(&cfg, &mut acct, &sell, dec!(50_000));
        let fee2 = sell_report.fee;

        assert_eq!(acct.total_fees_paid, fee1 + fee2);
    }

    #[test]
    fn zero_quantity_buy_fills_with_zero() {
        let cfg = test_config();
        let mut acct = VirtualAccount::new(dec!(10_000));
        let order = buy_order(1, dec!(0), dec!(50_000));

        let report = simulate_fill(&cfg, &mut acct, &order, dec!(50_000));
        assert_eq!(report.status, FillStatus::Filled);
        assert_eq!(report.fee, dec!(0));
        assert_eq!(acct.quote_balance, dec!(10_000)); // no cost for zero qty
    }

    #[test]
    fn slippage_direction_correct_for_both_sides() {
        let cfg = test_config();
        let market = dec!(100_000);

        let mut acct_buy = VirtualAccount::new(dec!(1_000_000));
        let buy = buy_order(1, dec!(1.0), market);
        let buy_report = simulate_fill(&cfg, &mut acct_buy, &buy, market);

        let mut acct_sell = VirtualAccount::new(dec!(0));
        acct_sell.credit_base("BTC", dec!(1.0));
        let sell = sell_order(1, dec!(1.0), market);
        let sell_report = simulate_fill(&cfg, &mut acct_sell, &sell, market);

        // Buy fill price > market (adverse slippage)
        assert!(buy_report.fill_price > market);
        // Sell fill price < market (adverse slippage)
        assert!(sell_report.fill_price < market);
    }
}
