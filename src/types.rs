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
