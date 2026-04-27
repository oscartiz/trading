//! # Virtual Account / Paper Engine
//!
//! Pillar 2: The local matching engine. Manages virtual balances, simulates
//! slippage and fee deduction, and produces execution reports.
//!
//! All financial arithmetic is performed exclusively with [`rust_decimal::Decimal`]
//! to guarantee deterministic base-10 calculations.

pub mod account;
pub mod matching;
