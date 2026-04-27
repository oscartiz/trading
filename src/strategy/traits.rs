//! Strategy trait and a reference momentum strategy implementation.
//!
//! The [`ExecutionClient`] trait is the boundary contract. Strategies consume
//! market data and emit order requests through it. They MUST NOT know whether
//! the downstream is a paper engine or a live exchange — this is enforced by
//! the trait abstraction.
//!
//! In the future, strategies in this module will execute local ML inference
//! (e.g. forward passes via `candle-core`) on price-action features before
//! deciding to trade. The trait surface remains identical.

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════
// Trait definition
// ═══════════════════════════════════════════════════════════════════════════

/// The execution boundary. Any strategy interacts with the outside world
/// exclusively through this trait.
#[async_trait]
pub trait ExecutionClient: Send + Sync {
    /// Submit an order request to the downstream execution venue.
    async fn submit_order(&self, order: OrderRequest) -> Result<(), crate::error::BotError>;
}

/// Channel-backed execution client that routes orders through an MPSC sender.
/// Whether the receiver is the paper engine or a live REST/WS endpoint is
/// irrelevant to the strategy.
pub struct ChannelExecutionClient {
    order_tx: mpsc::Sender<OrderRequest>,
}

impl ChannelExecutionClient {
    pub fn new(order_tx: mpsc::Sender<OrderRequest>) -> Self {
        Self { order_tx }
    }
}

#[async_trait]
impl ExecutionClient for ChannelExecutionClient {
    async fn submit_order(&self, order: OrderRequest) -> Result<(), crate::error::BotError> {
        self.order_tx
            .send(order)
            .await
            .map_err(|_| crate::error::BotError::ChannelClosed)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference strategy: simple momentum crossover
// ═══════════════════════════════════════════════════════════════════════════

/// A minimal momentum strategy that tracks a fast and slow EMA and fires
/// orders on crossover events. This exists purely as scaffolding — swap it
/// out for a candle-core inference strategy when ready.
pub async fn run_strategy(
    client: impl ExecutionClient,
    mut tick_rx: mpsc::Receiver<MarketTick>,
    mut report_rx: mpsc::Receiver<ExecutionReport>,
) {
    let mut ema_fast: Option<Decimal> = None;
    let mut ema_slow: Option<Decimal> = None;
    let mut prev_signal: Option<Side> = None;
    let mut order_id_counter: u64 = 0;

    // EMA smoothing factors (fast ≈ 12-period, slow ≈ 26-period)
    let alpha_fast = dec!(0.15);
    let alpha_slow = dec!(0.07);

    info!("Strategy task started — waiting for ticks…");

    loop {
        tokio::select! {
            Some(tick) = tick_rx.recv() => {
                // Update EMAs
                let fast = match ema_fast {
                    Some(prev) => alpha_fast * tick.price + (dec!(1) - alpha_fast) * prev,
                    None => tick.price,
                };
                let slow = match ema_slow {
                    Some(prev) => alpha_slow * tick.price + (dec!(1) - alpha_slow) * prev,
                    None => tick.price,
                };
                ema_fast = Some(fast);
                ema_slow = Some(slow);

                // Determine signal
                let signal = if fast > slow { Side::Buy } else { Side::Sell };

                // Only fire on crossover transitions
                if prev_signal.map_or(true, |prev| prev != signal) {
                    order_id_counter += 1;

                    // Fixed small size for paper trading
                    let qty = dec!(0.001);

                    let order = OrderRequest {
                        id: order_id_counter,
                        symbol: tick.symbol.clone(),
                        side: signal,
                        qty,
                        price: tick.price,
                    };

                    info!(
                        id = order.id,
                        side = ?signal,
                        price = %tick.price,
                        fast_ema = %fast,
                        slow_ema = %slow,
                        "Signal crossover → submitting order"
                    );

                    if let Err(e) = client.submit_order(order).await {
                        warn!(error = %e, "Failed to submit order");
                    }
                }

                prev_signal = Some(signal);
            }

            Some(report) = report_rx.recv() => {
                info!(
                    order_id = report.order_id,
                    status = ?report.status,
                    fill_price = %report.fill_price,
                    fee = %report.fee,
                    "Received execution report"
                );
            }

            else => {
                info!("All channels closed — strategy shutting down.");
                break;
            }
        }
    }
}
