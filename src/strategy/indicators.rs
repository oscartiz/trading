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

    // ═══════════════════════════════════════════════════════════════════
    // PriceBuffer
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn buffer_new_is_empty() {
        let buf = PriceBuffer::new(10);
        assert_eq!(buf.len(), 0);
        assert!(!buf.is_full());
        assert!(buf.last().is_none());
    }

    #[test]
    fn buffer_push_increments_length() {
        let mut buf = PriceBuffer::new(5);
        buf.push(dec!(100));
        assert_eq!(buf.len(), 1);
        buf.push(dec!(200));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn buffer_evicts_oldest_at_capacity() {
        let mut buf = PriceBuffer::new(3);
        buf.push(dec!(10));
        buf.push(dec!(20));
        buf.push(dec!(30));
        assert!(buf.is_full());
        assert_eq!(buf.len(), 3);

        // Push a 4th — should evict 10
        buf.push(dec!(40));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.prices(), &[dec!(20), dec!(30), dec!(40)]);
    }

    #[test]
    fn buffer_last_returns_most_recent() {
        let mut buf = PriceBuffer::new(5);
        buf.push(dec!(100));
        buf.push(dec!(200));
        assert_eq!(buf.last(), Some(dec!(200)));
    }

    #[test]
    fn buffer_is_full_exact_capacity() {
        let mut buf = PriceBuffer::new(2);
        assert!(!buf.is_full());
        buf.push(dec!(1));
        assert!(!buf.is_full());
        buf.push(dec!(2));
        assert!(buf.is_full());
    }

    // ═══════════════════════════════════════════════════════════════════
    // RSI
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn rsi_needs_minimum_data() {
        let prices = vec![dec!(100), dec!(101)];
        assert!(rsi(&prices, 14).is_none());
    }

    #[test]
    fn rsi_exact_minimum_data() {
        // period=14 needs exactly 15 prices (period + 1)
        let prices: Vec<Decimal> = (0..15).map(|i| Decimal::from(100 + i)).collect();
        assert!(rsi(&prices, 14).is_some());
    }

    #[test]
    fn rsi_fully_bullish_is_100() {
        // Monotonically increasing → RSI = 100 (no losses)
        let prices: Vec<Decimal> = (0..30).map(|i| Decimal::from(100 + i)).collect();
        let result = rsi(&prices, 14).unwrap();
        assert_eq!(result, dec!(100));
    }

    #[test]
    fn rsi_fully_bearish_is_0() {
        // Monotonically decreasing → RSI = 0 (no gains)
        let prices: Vec<Decimal> = (0..30).map(|i| Decimal::from(200 - i)).collect();
        let result = rsi(&prices, 14).unwrap();
        assert_eq!(result, dec!(0));
    }

    #[test]
    fn rsi_flat_market_is_midpoint() {
        // Alternating up/down of equal magnitude → RSI ≈ 50
        let mut prices = vec![dec!(100)];
        for i in 0..30 {
            if i % 2 == 0 {
                prices.push(prices.last().unwrap() + dec!(1));
            } else {
                prices.push(prices.last().unwrap() - dec!(1));
            }
        }
        let result = rsi(&prices, 14).unwrap();
        // Should be close to 50 (within a reasonable range due to Wilder smoothing)
        assert!(result > dec!(40) && result < dec!(60),
            "RSI for alternating market should be near 50, got {result}");
    }

    #[test]
    fn rsi_bounded_0_to_100() {
        // Test with various price patterns
        let patterns: Vec<Vec<Decimal>> = vec![
            (0..50).map(|i| Decimal::from(100 + i * 3)).collect(),      // strong uptrend
            (0..50).map(|i| Decimal::from(200 - i * 2)).collect(),      // strong downtrend
            vec![dec!(100); 50],                                         // flat (edge case)
            (0..50).map(|i| Decimal::from(100) + Decimal::from(i % 5)).collect(), // choppy
        ];

        for (idx, prices) in patterns.iter().enumerate() {
            if let Some(val) = rsi(prices, 14) {
                assert!(val >= dec!(0) && val <= dec!(100),
                    "Pattern {idx}: RSI out of bounds: {val}");
            }
        }
    }

    #[test]
    fn rsi_different_periods() {
        let prices: Vec<Decimal> = (0..100).map(|i| Decimal::from(100 + i)).collect();
        // All should return 100 for monotonically increasing data
        assert_eq!(rsi(&prices, 5).unwrap(), dec!(100));
        assert_eq!(rsi(&prices, 14).unwrap(), dec!(100));
        assert_eq!(rsi(&prices, 21).unwrap(), dec!(100));
    }

    // ═══════════════════════════════════════════════════════════════════
    // SMA
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn sma_insufficient_data() {
        let prices = vec![dec!(10), dec!(20)];
        assert!(sma(&prices, 5).is_none());
    }

    #[test]
    fn sma_basic_average() {
        let prices = vec![dec!(10), dec!(20), dec!(30)];
        assert_eq!(sma(&prices, 3).unwrap(), dec!(20));
    }

    #[test]
    fn sma_uses_last_n_prices() {
        let prices = vec![dec!(1), dec!(2), dec!(3), dec!(10), dec!(20), dec!(30)];
        // SMA-3 should average only the last 3: (10+20+30)/3 = 20
        assert_eq!(sma(&prices, 3).unwrap(), dec!(20));
    }

    #[test]
    fn sma_window_1_equals_last_price() {
        let prices = vec![dec!(10), dec!(20), dec!(42)];
        assert_eq!(sma(&prices, 1).unwrap(), dec!(42));
    }

    #[test]
    fn sma_flat_market() {
        let prices = vec![dec!(100); 20];
        assert_eq!(sma(&prices, 10).unwrap(), dec!(100));
    }

    #[test]
    fn sma_exact_minimum_data() {
        let prices = vec![dec!(5), dec!(10), dec!(15)];
        assert_eq!(sma(&prices, 3).unwrap(), dec!(10));
        assert!(sma(&prices, 4).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Volatility MAD
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn volatility_insufficient_data() {
        let prices = vec![dec!(100), dec!(101)];
        assert!(volatility_mad(&prices, 5).is_none());
    }

    #[test]
    fn volatility_flat_market_is_zero() {
        let prices = vec![dec!(100); 25];
        let vol = volatility_mad(&prices, 14).unwrap();
        assert_eq!(vol, dec!(0));
    }

    #[test]
    fn volatility_positive_for_moving_market() {
        // Steadily increasing prices → non-zero volatility
        let prices: Vec<Decimal> = (0..25).map(|i| Decimal::from(100 + i)).collect();
        let vol = volatility_mad(&prices, 14).unwrap();
        assert!(vol > dec!(0), "Volatility should be positive for trending market");
    }

    #[test]
    fn volatility_higher_for_choppy_market() {
        // Smooth uptrend
        let smooth: Vec<Decimal> = (0..25).map(|i| Decimal::from(100 + i)).collect();
        let vol_smooth = volatility_mad(&smooth, 14).unwrap();

        // Choppy market with bigger swings
        let mut choppy = Vec::new();
        for i in 0..25 {
            if i % 2 == 0 { choppy.push(Decimal::from(100 + i * 5)); }
            else { choppy.push(Decimal::from(100 - i * 3)); }
        }
        let vol_choppy = volatility_mad(&choppy, 14).unwrap();

        assert!(vol_choppy > vol_smooth,
            "Choppy market vol ({vol_choppy}) should exceed smooth ({vol_smooth})");
    }

    #[test]
    fn volatility_ignores_zero_prices() {
        // If a zero price appears, the return is skipped (div by zero guard)
        let mut prices = vec![dec!(0)];
        for i in 1..25 {
            prices.push(Decimal::from(100 + i));
        }
        // Should not panic
        let result = volatility_mad(&prices, 14);
        assert!(result.is_some());
    }
}

