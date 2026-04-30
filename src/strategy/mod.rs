//! # Strategy Module
//!
//! Pillar 3: The decoupled strategy interface. Strategies implement
//! [`ExecutionClient`] and are entirely unaware of whether they communicate
//! with the paper engine or a live execution endpoint.
//!
//! Available strategies:
//! - `dca_accumulator` — Conservative DCA with RSI/SMA filters and risk management
//! - `traits` — Core trait definitions and the reference EMA crossover strategy

pub mod deeplob_maker;
pub mod ml;
pub mod risk;
pub mod traits;
