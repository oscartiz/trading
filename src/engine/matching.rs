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
    mut tick_rx: mpsc::Receiver<MarketTick>,
) {
    let mut account = VirtualAccount::new(config.initial_balance);
    let mut last_price: Option<Decimal> = None;

    info!("Paper engine started.");

    loop {
        tokio::select! {
            // Continuously update the latest market price
            Some(tick) = tick_rx.recv() => {
                last_price = Some(tick.price);
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
fn simulate_fill(
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
