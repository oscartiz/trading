//! Risk management module.
//!
//! Risk controls: max allocation, per-trade limit, drawdown circuit breaker,
//! cooldown, volatility scaling, trailing stop-loss. No leverage, no shorts.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct RiskParams {
    pub max_allocation: Decimal,
    pub max_trade_notional: Decimal,
    pub base_trade_notional: Decimal,
    pub max_drawdown_pct: Decimal,
    pub cooldown_secs: u64,
    pub high_vol_threshold: Decimal,
    pub extreme_vol_threshold: Decimal,
    pub trailing_stop_pct: Decimal,
}

impl Default for RiskParams {
    fn default() -> Self {
        Self {
            max_allocation: dec!(0.50),
            max_trade_notional: dec!(200),
            base_trade_notional: dec!(50),
            max_drawdown_pct: dec!(0.15),
            cooldown_secs: 300,
            high_vol_threshold: dec!(0.025),
            extreme_vol_threshold: dec!(0.050),
            trailing_stop_pct: dec!(0.12),
        }
    }
}

#[derive(Debug)]
pub struct RiskManager {
    pub params: RiskParams,
    peak_portfolio_value: Decimal,
    last_trade_ts: u64,
    position_peak_price: Decimal,
    circuit_breaker_active: bool,
}

impl RiskManager {
    pub fn new(params: RiskParams, initial_balance: Decimal) -> Self {
        Self {
            params,
            peak_portfolio_value: initial_balance,
            last_trade_ts: 0,
            position_peak_price: dec!(0),
            circuit_breaker_active: false,
        }
    }

    pub fn update_portfolio_value(&mut self, portfolio_value: Decimal) {
        if portfolio_value > self.peak_portfolio_value {
            self.peak_portfolio_value = portfolio_value;
            if self.circuit_breaker_active {
                info!(peak = %self.peak_portfolio_value, "Circuit breaker reset");
                self.circuit_breaker_active = false;
            }
        }
        if self.peak_portfolio_value > dec!(0) {
            let dd = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value;
            if dd >= self.params.max_drawdown_pct && !self.circuit_breaker_active {
                warn!(drawdown = %dd, "CIRCUIT BREAKER ACTIVATED");
                self.circuit_breaker_active = true;
            }
        }
    }

    pub fn update_price_peak(&mut self, price: Decimal) {
        if price > self.position_peak_price {
            self.position_peak_price = price;
        }
    }

    pub fn trailing_stop_triggered(&self, price: Decimal) -> bool {
        if self.position_peak_price == dec!(0) { return false; }
        let decline = (self.position_peak_price - price) / self.position_peak_price;
        decline >= self.params.trailing_stop_pct
    }

    pub fn approve_buy(
        &mut self, ts: u64, quote_bal: Decimal, crypto_val: Decimal, vol: Option<Decimal>,
    ) -> Option<Decimal> {
        let pv = quote_bal + crypto_val;
        if self.circuit_breaker_active { return None; }
        if ts.saturating_sub(self.last_trade_ts) < self.params.cooldown_secs * 1000 { return None; }
        let alloc = if pv > dec!(0) { crypto_val / pv } else { dec!(0) };
        if alloc >= self.params.max_allocation { return None; }
        let headroom = pv * self.params.max_allocation - crypto_val;
        let mut notional = self.params.base_trade_notional;
        if let Some(v) = vol {
            if v >= self.params.extreme_vol_threshold { return None; }
            if v >= self.params.high_vol_threshold { notional /= dec!(2); }
        }
        notional = notional.min(self.params.max_trade_notional).min(headroom).min(quote_bal);
        if notional <= dec!(0) { None } else { Some(notional) }
    }

    pub fn record_trade(&mut self, ts: u64) { self.last_trade_ts = ts; }
    pub fn reset_trailing_stop(&mut self) { self.position_peak_price = dec!(0); }
    pub fn is_circuit_breaker_active(&self) -> bool { self.circuit_breaker_active }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rm() -> RiskManager {
        RiskManager::new(RiskParams::default(), dec!(10_000))
    }

    // ═══════════════════════════════════════════════════════════════════
    // Circuit Breaker
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn circuit_breaker_inactive_on_init() {
        let rm = default_rm();
        assert!(!rm.is_circuit_breaker_active());
    }

    #[test]
    fn circuit_breaker_triggers_at_15pct_drawdown() {
        let mut rm = default_rm();
        // 15% drawdown from 10k peak → should trigger
        rm.update_portfolio_value(dec!(8500));
        assert!(rm.is_circuit_breaker_active());
    }

    #[test]
    fn circuit_breaker_does_not_trigger_below_threshold() {
        let mut rm = default_rm();
        // 14.99% drawdown → should NOT trigger
        rm.update_portfolio_value(dec!(8501));
        assert!(!rm.is_circuit_breaker_active());
    }

    #[test]
    fn circuit_breaker_resets_on_new_high() {
        let mut rm = default_rm();
        rm.update_portfolio_value(dec!(8000)); // trigger
        assert!(rm.is_circuit_breaker_active());

        rm.update_portfolio_value(dec!(10_001)); // new ATH
        assert!(!rm.is_circuit_breaker_active());
    }

    #[test]
    fn circuit_breaker_blocks_all_buys() {
        let mut rm = default_rm();
        rm.update_portfolio_value(dec!(8000)); // trigger

        let result = rm.approve_buy(999_999_999, dec!(8000), dec!(0), None);
        assert!(result.is_none(), "Buys should be blocked when circuit breaker is active");
    }

    #[test]
    fn circuit_breaker_progressive_drawdown() {
        let mut rm = default_rm();
        // Portfolio gradually declines
        rm.update_portfolio_value(dec!(9500)); // 5% dd — no trigger
        assert!(!rm.is_circuit_breaker_active());
        rm.update_portfolio_value(dec!(9000)); // 10% dd — no trigger
        assert!(!rm.is_circuit_breaker_active());
        rm.update_portfolio_value(dec!(8500)); // 15% dd — trigger!
        assert!(rm.is_circuit_breaker_active());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Cooldown
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn cooldown_blocks_rapid_trades() {
        let mut rm = default_rm();
        rm.record_trade(1_000_000);

        // 1 second later → blocked (cooldown = 300s = 300_000ms)
        let result = rm.approve_buy(1_001_000, dec!(9000), dec!(1000), None);
        assert!(result.is_none());
    }

    #[test]
    fn cooldown_allows_after_expiry() {
        let mut rm = default_rm();
        rm.record_trade(1_000_000);

        // 301 seconds later → allowed
        let result = rm.approve_buy(1_301_000, dec!(9000), dec!(1000), None);
        assert!(result.is_some());
    }

    #[test]
    fn cooldown_exact_boundary() {
        let mut rm = default_rm();
        rm.record_trade(1_000_000);

        // Exactly 300 seconds → should pass (300_000ms elapsed = cooldown)
        let result = rm.approve_buy(1_300_000, dec!(9000), dec!(1000), None);
        assert!(result.is_some());
    }

    #[test]
    fn first_trade_has_no_cooldown() {
        let mut rm = default_rm();
        // last_trade_ts starts at 0, so any reasonable timestamp passes
        let result = rm.approve_buy(1_000_000, dec!(10_000), dec!(0), None);
        assert!(result.is_some());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Max Allocation
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn blocks_buy_at_max_allocation() {
        let mut rm = default_rm();
        // Portfolio: 5k quote + 5k crypto = 50% allocation → blocked
        let result = rm.approve_buy(999_999_999, dec!(5000), dec!(5000), None);
        assert!(result.is_none());
    }

    #[test]
    fn blocks_buy_over_max_allocation() {
        let mut rm = default_rm();
        // 60% in crypto → blocked
        let result = rm.approve_buy(999_999_999, dec!(4000), dec!(6000), None);
        assert!(result.is_none());
    }

    #[test]
    fn allows_buy_below_max_allocation() {
        let mut rm = default_rm();
        // 10% in crypto → allowed
        let result = rm.approve_buy(999_999_999, dec!(9000), dec!(1000), None);
        assert!(result.is_some());
    }

    #[test]
    fn buy_capped_to_allocation_headroom() {
        let mut rm = default_rm();
        // Portfolio: 6k quote + 4k crypto = 40% allocation
        // Headroom = 10k * 0.50 - 4k = 1k
        // base_trade_notional = 50, which is < headroom, so not capped here
        let result = rm.approve_buy(999_999_999, dec!(6000), dec!(4000), None);
        assert!(result.is_some());
        let notional = result.unwrap();
        assert!(notional <= dec!(50), "Should be base notional since headroom > base");
    }

    #[test]
    fn buy_capped_when_near_limit() {
        let mut rm = default_rm();
        rm.params.base_trade_notional = dec!(500);
        // Portfolio: 5100 quote + 4900 crypto = 49% allocation
        // Headroom = 10k * 0.50 - 4900 = 100
        let result = rm.approve_buy(999_999_999, dec!(5100), dec!(4900), None);
        assert!(result.is_some());
        let notional = result.unwrap();
        assert_eq!(notional, dec!(100), "Should be capped to allocation headroom");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Volatility Scaling
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn high_vol_halves_position_size() {
        let mut rm = default_rm();
        let normal = rm.approve_buy(999_999_999, dec!(9000), dec!(1000), None).unwrap();

        let mut rm2 = default_rm();
        let high_vol = rm2.approve_buy(
            999_999_999, dec!(9000), dec!(1000), Some(dec!(0.030)),
        ).unwrap();

        assert_eq!(high_vol, normal / dec!(2),
            "High vol should halve the position size");
    }

    #[test]
    fn extreme_vol_blocks_buy() {
        let mut rm = default_rm();
        let result = rm.approve_buy(
            999_999_999, dec!(9000), dec!(1000), Some(dec!(0.060)),
        );
        assert!(result.is_none(), "Extreme vol should block all buys");
    }

    #[test]
    fn normal_vol_no_scaling() {
        let mut rm1 = default_rm();
        let no_vol = rm1.approve_buy(999_999_999, dec!(9000), dec!(1000), None).unwrap();

        let mut rm2 = default_rm();
        let low_vol = rm2.approve_buy(
            999_999_999, dec!(9000), dec!(1000), Some(dec!(0.010)),
        ).unwrap();

        assert_eq!(no_vol, low_vol, "Low vol should not affect size");
    }

    #[test]
    fn vol_at_exact_high_threshold() {
        let mut rm = default_rm();
        let result = rm.approve_buy(
            999_999_999, dec!(9000), dec!(1000), Some(dec!(0.025)),
        );
        assert!(result.is_some());
        // At threshold → halved
        assert_eq!(result.unwrap(), dec!(25));
    }

    #[test]
    fn vol_at_exact_extreme_threshold() {
        let mut rm = default_rm();
        let result = rm.approve_buy(
            999_999_999, dec!(9000), dec!(1000), Some(dec!(0.050)),
        );
        assert!(result.is_none(), "At exact extreme threshold should block");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Trailing Stop
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn trailing_stop_not_triggered_without_peak() {
        let rm = default_rm();
        // No peak set → should never trigger
        assert!(!rm.trailing_stop_triggered(dec!(50_000)));
    }

    #[test]
    fn trailing_stop_triggers_at_12pct_decline() {
        let mut rm = default_rm();
        rm.update_price_peak(dec!(100_000));
        // 12% decline from 100k = 88k
        assert!(rm.trailing_stop_triggered(dec!(88_000)));
    }

    #[test]
    fn trailing_stop_does_not_trigger_small_decline() {
        let mut rm = default_rm();
        rm.update_price_peak(dec!(100_000));
        // 5% decline — no trigger
        assert!(!rm.trailing_stop_triggered(dec!(95_000)));
    }

    #[test]
    fn trailing_stop_resets_correctly() {
        let mut rm = default_rm();
        rm.update_price_peak(dec!(100_000));
        assert!(rm.trailing_stop_triggered(dec!(85_000))); // triggers

        rm.reset_trailing_stop();
        assert!(!rm.trailing_stop_triggered(dec!(85_000))); // no peak → no trigger
    }

    #[test]
    fn trailing_stop_tracks_highest_price() {
        let mut rm = default_rm();
        rm.update_price_peak(dec!(90_000));
        rm.update_price_peak(dec!(100_000));
        rm.update_price_peak(dec!(95_000)); // lower — ignored

        // 12% of 100k (the peak) = 88k
        assert!(!rm.trailing_stop_triggered(dec!(89_000))); // above 88k
        assert!(rm.trailing_stop_triggered(dec!(87_000)));  // below 88k
    }

    // ═══════════════════════════════════════════════════════════════════
    // Per-Trade & Cash Limits
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn buy_capped_to_max_trade_notional() {
        let mut rm = default_rm();
        rm.params.base_trade_notional = dec!(500); // higher than max
        rm.params.max_trade_notional = dec!(200);

        let result = rm.approve_buy(999_999_999, dec!(9000), dec!(1000), None).unwrap();
        assert_eq!(result, dec!(200));
    }

    #[test]
    fn buy_capped_to_available_cash() {
        let mut rm = default_rm();
        // Only $30 in quote balance, $0 crypto
        // Headroom = $30 * 0.50 = $15, but base = $50, so capped to $15
        // Then capped to min($15, $30 cash) = $15
        let result = rm.approve_buy(999_999_999, dec!(30), dec!(0), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), dec!(15));
    }

    #[test]
    fn zero_cash_blocks_buy() {
        let mut rm = default_rm();
        let result = rm.approve_buy(999_999_999, dec!(0), dec!(5000), None);
        assert!(result.is_none());
    }

    #[test]
    fn zero_portfolio_value_blocks_buy() {
        let mut rm = default_rm();
        let result = rm.approve_buy(999_999_999, dec!(0), dec!(0), None);
        assert!(result.is_none());
    }
}
