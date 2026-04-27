//! Virtual account state — balances and position tracking.

use std::collections::HashMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::info;

/// The virtual account that tracks quote-currency (USDT) and
/// base-currency holdings per symbol.
#[derive(Debug)]
pub struct VirtualAccount {
    /// Available quote balance (e.g. USDT).
    pub quote_balance: Decimal,
    /// Holdings of each base asset, keyed by uppercase symbol (e.g. "BTC").
    pub holdings: HashMap<String, Decimal>,
    /// Cumulative fees paid.
    pub total_fees_paid: Decimal,
    /// Cumulative realized P&L.
    pub realized_pnl: Decimal,
}

impl VirtualAccount {
    pub fn new(initial_quote: Decimal) -> Self {
        info!(balance = %initial_quote, "Initializing virtual account");
        Self {
            quote_balance: initial_quote,
            holdings: HashMap::new(),
            total_fees_paid: dec!(0),
            realized_pnl: dec!(0),
        }
    }

    /// Return the base asset string from a pair symbol.
    /// E.g. "BTCUSDT" → "BTC"
    pub fn base_asset(symbol: &str) -> &str {
        symbol.strip_suffix("USDT").unwrap_or(symbol)
    }

    /// Current holding of a given base asset.
    pub fn holding(&self, base: &str) -> Decimal {
        self.holdings.get(base).copied().unwrap_or(dec!(0))
    }

    /// Credit base asset.
    pub fn credit_base(&mut self, base: &str, qty: Decimal) {
        *self.holdings.entry(base.to_string()).or_insert(dec!(0)) += qty;
    }

    /// Debit base asset. Caller must ensure sufficient balance.
    pub fn debit_base(&mut self, base: &str, qty: Decimal) {
        let entry = self.holdings.entry(base.to_string()).or_insert(dec!(0));
        *entry -= qty;
    }
}
