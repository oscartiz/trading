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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_account_has_correct_balance() {
        let acct = VirtualAccount::new(dec!(10_000));
        assert_eq!(acct.quote_balance, dec!(10_000));
        assert_eq!(acct.total_fees_paid, dec!(0));
        assert_eq!(acct.realized_pnl, dec!(0));
        assert!(acct.holdings.is_empty());
    }

    #[test]
    fn base_asset_strips_usdt_suffix() {
        assert_eq!(VirtualAccount::base_asset("BTCUSDT"), "BTC");
        assert_eq!(VirtualAccount::base_asset("ETHUSDT"), "ETH");
        assert_eq!(VirtualAccount::base_asset("SOLUSDT"), "SOL");
    }

    #[test]
    fn base_asset_preserves_non_usdt() {
        assert_eq!(VirtualAccount::base_asset("BTCETH"), "BTCETH");
        assert_eq!(VirtualAccount::base_asset("BTC"), "BTC");
    }

    #[test]
    fn holding_returns_zero_for_unknown() {
        let acct = VirtualAccount::new(dec!(10_000));
        assert_eq!(acct.holding("BTC"), dec!(0));
        assert_eq!(acct.holding("NONEXISTENT"), dec!(0));
    }

    #[test]
    fn credit_base_adds_to_holdings() {
        let mut acct = VirtualAccount::new(dec!(10_000));
        acct.credit_base("BTC", dec!(0.5));
        assert_eq!(acct.holding("BTC"), dec!(0.5));

        acct.credit_base("BTC", dec!(0.3));
        assert_eq!(acct.holding("BTC"), dec!(0.8));
    }

    #[test]
    fn debit_base_subtracts_from_holdings() {
        let mut acct = VirtualAccount::new(dec!(10_000));
        acct.credit_base("BTC", dec!(1.0));
        acct.debit_base("BTC", dec!(0.3));
        assert_eq!(acct.holding("BTC"), dec!(0.7));
    }

    #[test]
    fn multiple_assets_tracked_independently() {
        let mut acct = VirtualAccount::new(dec!(10_000));
        acct.credit_base("BTC", dec!(0.5));
        acct.credit_base("ETH", dec!(10.0));

        assert_eq!(acct.holding("BTC"), dec!(0.5));
        assert_eq!(acct.holding("ETH"), dec!(10.0));

        acct.debit_base("BTC", dec!(0.1));
        assert_eq!(acct.holding("BTC"), dec!(0.4));
        assert_eq!(acct.holding("ETH"), dec!(10.0)); // unchanged
    }

    #[test]
    fn full_debit_zeroes_holding() {
        let mut acct = VirtualAccount::new(dec!(10_000));
        acct.credit_base("BTC", dec!(1.0));
        acct.debit_base("BTC", dec!(1.0));
        assert_eq!(acct.holding("BTC"), dec!(0));
    }
}
