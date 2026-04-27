//! # Market Data Feed
//!
//! Pillar 1: Manages a resilient WebSocket connection to Binance's
//! aggregate trade stream. Deserializes raw JSON frames into [`MarketTick`]s
//! and fans them out to downstream consumers via MPSC channels.

pub mod binance;
