//! # DeepLOB HFT Bot
//!
//! Entry point. Initializes the tokio runtime, configures structured logging,
//! plumbs the MPSC channels connecting all pillars, and spawns each
//! as an independent async task.

mod config;
mod dashboard;
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
        "Starting DeepLOB autonomous trading bot"
    );

    // ── Channel Plumbing ────────────────────────────────────────────────
    let (tick_tx_strategy, tick_rx_strategy) =
        mpsc::channel::<types::MarketEvent>(config.channel_buffer);
    let (tick_tx_engine, tick_rx_engine) =
        mpsc::channel::<types::MarketEvent>(config.channel_buffer);

    // Strategy → Engine: order requests
    let (order_tx, order_rx) = mpsc::channel::<types::OrderRequest>(config.channel_buffer);

    // Engine → Strategy: execution reports
    let (report_tx, report_rx) = mpsc::channel::<types::ExecutionReport>(config.channel_buffer);

    // Strategy → Dashboard: portfolio snapshots
    let (snapshot_tx, snapshot_rx) = mpsc::channel::<types::PortfolioSnapshot>(256);

    // ── Spawn Pillar 4: Dashboard Monitor ───────────────────────────────
    let dashboard_port = 3030;
    let dashboard_handle = tokio::spawn(async move {
        dashboard::server::run_dashboard(snapshot_rx, dashboard_port).await;
    });

    // ── Spawn Pillar 1: Market Data Feed ────────────────────────────────
    let feed_config = config.clone();
    let feed_handle = tokio::spawn(async move {
        let (internal_tx, mut internal_rx) =
            mpsc::channel::<types::MarketEvent>(feed_config.channel_buffer);

        let tx_s = tick_tx_strategy;
        let tx_e = tick_tx_engine;

        let fanout = tokio::spawn(async move {
            while let Some(tick) = internal_rx.recv().await {
                let t1 = tx_s.send(tick.clone());
                let t2 = tx_e.send(tick);
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

    // ── Spawn Pillar 3: Strategy (DeepLOB Maker) ───────────────────────────
    let execution_client = ChannelExecutionClient::new(order_tx);
    let initial_balance = config.initial_balance;
    let strategy_handle = tokio::spawn(async move {
        strategy::deeplob_maker::run_deeplob_strategy(
            execution_client,
            tick_rx_strategy,
            report_rx,
            initial_balance,
            Some(snapshot_tx),
        )
        .await;
    });

    info!(
        dashboard = format!("http://localhost:{dashboard_port}"),
        "DeepLOB HFT bot initialized successfully"
    );

    // ── Await Completion ────────────────────────────────────────────────
    tokio::select! {
        _ = feed_handle => info!("Feed task exited."),
        _ = engine_handle => info!("Engine task exited."),
        _ = strategy_handle => info!("Strategy task exited."),
        _ = dashboard_handle => info!("Dashboard task exited."),
    }

    info!("Bot shutdown complete.");
}
