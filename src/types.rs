//! Shared message types that flow through MPSC channels.
//!
//! These are the *only* coupling point between pillars. Each struct is
//! designed to be cheaply cloneable and `Send + Sync` for channel transport.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// Market Data → Strategy (and Engine for price updates)
// ═══════════════════════════════════════════════════════════════════════════

/// Normalized tick arriving from the Binance stream.
/// Deserialized from the `@aggTrade` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    /// Trading pair, e.g. "BTCUSDT"
    pub symbol: String,
    /// Aggregate trade price — always base-10 via rust_decimal
    #[serde(with = "rust_decimal::serde::str")]
    pub price: Decimal,
    /// Aggregate trade quantity
    #[serde(with = "rust_decimal::serde::str")]
    pub qty: Decimal,
    /// Event timestamp (ms since epoch)
    pub timestamp: u64,
    /// Was the buyer the maker?
    pub is_buyer_maker: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Strategy → Engine (order requests)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// An order request emitted by the strategy.
/// The engine will simulate fill, slippage, and fees.
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub id: u64,
    pub symbol: String,
    pub side: Side,
    pub qty: Decimal,
    /// Limit price hint (the engine applies simulated slippage on top).
    pub price: Decimal,
}

// ═══════════════════════════════════════════════════════════════════════════
// Engine → Strategy (execution reports)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillStatus {
    Filled,
    PartialFill,
    Rejected,
}

/// Execution report sent back from the paper engine after processing
/// an [`OrderRequest`].
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub order_id: u64,
    pub side: Side,
    pub status: FillStatus,
    pub filled_qty: Decimal,
    pub fill_price: Decimal,
    pub fee: Decimal,
}

// ═══════════════════════════════════════════════════════════════════════════
// Strategy → Dashboard (portfolio snapshots)
// ═══════════════════════════════════════════════════════════════════════════

/// A point-in-time snapshot of portfolio state, streamed to the dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioSnapshot {
    /// Timestamp in ms since epoch
    pub timestamp: u64,
    /// Current BTC price
    #[serde(with = "rust_decimal::serde::str")]
    pub price: Decimal,
    /// USDT cash balance
    #[serde(with = "rust_decimal::serde::str")]
    pub quote_balance: Decimal,
    /// BTC quantity held
    #[serde(with = "rust_decimal::serde::str")]
    pub crypto_qty: Decimal,
    /// Total portfolio value (quote + crypto * price)
    #[serde(with = "rust_decimal::serde::str")]
    pub portfolio_value: Decimal,
    /// Crypto allocation as percentage
    #[serde(with = "rust_decimal::serde::str")]
    pub allocation_pct: Decimal,
    /// Cumulative cost basis
    #[serde(with = "rust_decimal::serde::str")]
    pub cost_basis: Decimal,
    /// Unrealized P&L
    #[serde(with = "rust_decimal::serde::str")]
    pub unrealized_pnl: Decimal,
    /// Current RSI (if available)
    pub rsi: Option<f64>,
    /// Recent trade events history
    pub event_history: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Dashboard → Strategy (manual commands)
// ═══════════════════════════════════════════════════════════════════════════

/// Manual commands sent from the dashboard UI to the strategy.
#[derive(Debug, Clone)]
pub enum DashboardCommand {
    /// Manual market buy with a specific USDT notional amount
    ManualBuy { notional: Decimal },
    /// Panic sell — liquidate entire position immediately
    PanicSell,
}
