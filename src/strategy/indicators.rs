//! Technical indicators computed over a rolling window of prices.
//!
//! All calculations use `rust_decimal::Decimal` for deterministic base-10 math.
//! These are pure functions with no side effects — they operate on price buffers
//! and return indicator values.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// ═══════════════════════════════════════════════════════════════════════════
// Rolling price buffer
// ═══════════════════════════════════════════════════════════════════════════

/// A fixed-capacity circular buffer that retains the last `N` prices.
/// Used as the input to all indicator calculations.
#[derive(Debug)]
pub struct PriceBuffer {
    buf: Vec<Decimal>,
    capacity: usize,
}

impl PriceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new price, evicting the oldest if at capacity.
    pub fn push(&mut self, price: Decimal) {
        if self.buf.len() == self.capacity {
            self.buf.remove(0);
        }
        self.buf.push(price);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_full(&self) -> bool {
        self.buf.len() == self.capacity
    }

    pub fn prices(&self) -> &[Decimal] {
        &self.buf
    }

    /// Return the most recent price, if any.
    pub fn last(&self) -> Option<Decimal> {
        self.buf.last().copied()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RSI (Relative Strength Index)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the 14-period RSI using the Wilder smoothing method.
///
/// Returns `None` if the buffer has fewer than `period + 1` prices.
/// RSI range: 0–100.
///   - RSI > 70 → overbought (avoid buying)
///   - RSI < 30 → oversold   (favorable entry)
pub fn rsi(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period + 1 {
        return None;
    }

    let mut avg_gain = dec!(0);
    let mut avg_loss = dec!(0);

    // Seed with simple average of first `period` changes
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > dec!(0) {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }

    let period_dec = Decimal::from(period as u64);
    avg_gain /= period_dec;
    avg_loss /= period_dec;

    // Wilder smoothing for remaining prices
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > dec!(0) {
            avg_gain = (avg_gain * (period_dec - dec!(1)) + change) / period_dec;
            avg_loss = (avg_loss * (period_dec - dec!(1))) / period_dec;
        } else {
            avg_gain = (avg_gain * (period_dec - dec!(1))) / period_dec;
            avg_loss = (avg_loss * (period_dec - dec!(1)) + change.abs()) / period_dec;
        }
    }

    if avg_loss == dec!(0) {
        return Some(dec!(100));
    }

    let rs = avg_gain / avg_loss;
    let rsi_value = dec!(100) - (dec!(100) / (dec!(1) + rs));
    Some(rsi_value)
}

// ═══════════════════════════════════════════════════════════════════════════
// Simple Moving Average
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the simple moving average over the last `window` prices.
pub fn sma(prices: &[Decimal], window: usize) -> Option<Decimal> {
    if prices.len() < window {
        return None;
    }
    let slice = &prices[prices.len() - window..];
    let sum: Decimal = slice.iter().copied().sum();
    Some(sum / Decimal::from(window as u64))
}

// ═══════════════════════════════════════════════════════════════════════════
// Volatility (standard deviation proxy)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the mean absolute deviation of returns over the last `window` prices.
/// This is a robust, Decimal-friendly volatility proxy (no sqrt needed).
///
/// Returns the MAD as a fraction of price — e.g. 0.015 = 1.5% average move.
pub fn volatility_mad(prices: &[Decimal], window: usize) -> Option<Decimal> {
    if prices.len() < window + 1 {
        return None;
    }

    let start = prices.len() - window - 1;
    let mut returns = Vec::with_capacity(window);

    for i in (start + 1)..prices.len() {
        if prices[i - 1] != dec!(0) {
            let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
            returns.push(ret);
        }
    }

    if returns.is_empty() {
        return Some(dec!(0));
    }

    let n = Decimal::from(returns.len() as u64);
    let mean: Decimal = returns.iter().copied().sum::<Decimal>() / n;
    let mad: Decimal = returns.iter().map(|r| (*r - mean).abs()).sum::<Decimal>() / n;

    Some(mad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn rsi_fully_bullish() {
        // Monotonically increasing prices → RSI should be 100
        let prices: Vec<Decimal> = (0..20).map(|i| Decimal::from(100 + i)).collect();
        let result = rsi(&prices, 14).unwrap();
        assert_eq!(result, dec!(100));
    }

    #[test]
    fn rsi_needs_minimum_data() {
        let prices = vec![dec!(100), dec!(101)];
        assert!(rsi(&prices, 14).is_none());
    }

    #[test]
    fn sma_basic() {
        let prices = vec![dec!(10), dec!(20), dec!(30)];
        assert_eq!(sma(&prices, 3).unwrap(), dec!(20));
    }

    #[test]
    fn volatility_flat_market() {
        let prices = vec![dec!(100); 20];
        let vol = volatility_mad(&prices, 14).unwrap();
        assert_eq!(vol, dec!(0));
    }
}
