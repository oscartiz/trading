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
