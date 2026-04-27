//! # Crypto Paper Trading Bot
//!
//! Entry point. Initializes the tokio runtime, configures structured logging,
//! plumbs the MPSC channels connecting all three pillars, and spawns each
//! as an independent async task.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────────┐       tick_tx        ┌──────────────────┐
//!  │  Binance WS Feed │ ──────────────────▶ │  Strategy Task    │
//!  │  (feed::binance) │ ──┐                 │  (strategy::run)  │
//!  └─────────────────┘   │                 └────────┬─────────┘
//!                         │  tick_tx_engine          │ order_tx
//!                         ▼                          ▼
//!                  ┌──────────────────┐     ┌──────────────────┐
//!                  │  Paper Engine     │◀────│  ChannelExec     │
//!                  │  (engine::run)    │     │  Client          │
//!                  └────────┬─────────┘     └──────────────────┘
//!                           │ report_tx
//!                           ▼
//!                    back to Strategy
//! ```

mod config;
mod engine;
mod error;
mod feed;
mod strategy;
mod types;

use config::Config;
use strategy::traits::ChannelExecutionClient;
use tokio::sync::mpsc;
use tracing::info;

#[tokio::main]
async fn main() {
    // ── Structured Logging ──────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    let config = Config::default();

    info!(
        symbol = %config.symbol,
        balance = %config.initial_balance,
        fee_rate = %config.fee_rate,
        slippage_bps = %config.slippage_bps,
        "Starting crypto paper trading bot"
    );

    // ── Channel Plumbing ────────────────────────────────────────────────
    //
    // Two tick channels: one for the strategy, one for the engine's price tracker.
    // This avoids a single consumer bottleneck and lets each pillar process
    // ticks at its own cadence.
    let (tick_tx_strategy, tick_rx_strategy) =
        mpsc::channel(config.channel_buffer);
    let (tick_tx_engine, tick_rx_engine) =
        mpsc::channel(config.channel_buffer);

    // Strategy → Engine: order requests
    let (order_tx, order_rx) = mpsc::channel::<types::OrderRequest>(config.channel_buffer);

    // Engine → Strategy: execution reports
    let (report_tx, report_rx) = mpsc::channel::<types::ExecutionReport>(config.channel_buffer);

    // ── Spawn Pillar 1: Market Data Feed ────────────────────────────────
    let feed_config = config.clone();
    let feed_handle = tokio::spawn(async move {
        // Fan-out: clone each tick to both the strategy and engine channels.
        let (internal_tx, mut internal_rx) =
            mpsc::channel::<types::MarketTick>(feed_config.channel_buffer);

        let tx_s = tick_tx_strategy;
        let tx_e = tick_tx_engine;

        // Fan-out task: reads from the single feed and duplicates to both consumers
        let fanout = tokio::spawn(async move {
            while let Some(tick) = internal_rx.recv().await {
                let t1 = tx_s.send(tick.clone());
                let t2 = tx_e.send(tick);
                // If either consumer drops, we still try the other
                let _ = tokio::join!(t1, t2);
            }
        });

        feed::binance::run_feed(feed_config, internal_tx).await;
        fanout.abort();
    });

    // ── Spawn Pillar 2: Paper Engine ────────────────────────────────────
    let engine_config = config.clone();
    let engine_handle = tokio::spawn(async move {
        engine::matching::run_engine(engine_config, order_rx, report_tx, tick_rx_engine).await;
    });

    // ── Spawn Pillar 3: Strategy (DCA Accumulator) ─────────────────────
    let execution_client = ChannelExecutionClient::new(order_tx);
    let initial_balance = config.initial_balance;
    let strategy_handle = tokio::spawn(async move {
        strategy::dca_accumulator::run_dca_strategy(
            execution_client,
            tick_rx_strategy,
            report_rx,
            initial_balance,
        )
        .await;
    });

    // ── Await Completion ────────────────────────────────────────────────
    // In production, you'd add signal handling (SIGINT/SIGTERM) for graceful
    // shutdown via tokio::signal. For now, run until any pillar exits.
    tokio::select! {
        _ = feed_handle => info!("Feed task exited."),
        _ = engine_handle => info!("Engine task exited."),
        _ = strategy_handle => info!("Strategy task exited."),
    }

    info!("Bot shutdown complete.");
}
