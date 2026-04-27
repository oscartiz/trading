//! Unified error types for the trading system.
//!
//! Each pillar defines its own error variant so channel receivers can
//! pattern-match on failure domain without downcasting.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BotError {
    // ── Feed Errors ─────────────────────────────────────────────────────
    #[error("WebSocket connection failed: {0}")]
    WsConnection(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("Stream deserialization failed: {0}")]
    Deserialize(#[from] serde_json::Error),

    // ── Engine Errors ───────────────────────────────────────────────────
    #[error("Insufficient balance: need {needed}, have {available}")]
    InsufficientBalance {
        needed: rust_decimal::Decimal,
        available: rust_decimal::Decimal,
    },

    #[error("Unknown symbol: {0}")]
    UnknownSymbol(String),

    // ── Channel Errors ──────────────────────────────────────────────────
    #[error("Channel closed unexpectedly")]
    ChannelClosed,

    // ── Catch-all ───────────────────────────────────────────────────────
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, BotError>;
