//! Runtime configuration for the trading bot.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Top-level configuration. In production you'd deserialize this from
/// a TOML/YAML file; for now we hard-code sensible paper-trading defaults.
#[derive(Debug, Clone)]
pub struct Config {
    /// Binance WebSocket base URL for the aggregated trade stream.
    pub ws_url: String,
    /// Trading pair to subscribe to.
    pub symbol: String,
    /// Starting virtual USDT balance.
    pub initial_balance: Decimal,
    /// Simulated taker fee (e.g. 0.001 = 10 bps).
    pub fee_rate: Decimal,
    /// Simulated slippage as a fraction of price (e.g. 0.0005 = 5 bps).
    pub slippage_bps: Decimal,
    /// Channel buffer depth — controls backpressure.
    pub channel_buffer: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ws_url: "wss://stream.binance.com:9443/ws".into(),
            symbol: "btcusdt".into(),
            initial_balance: dec!(10_000),
            fee_rate: dec!(0.001),
            slippage_bps: dec!(0.0005),
            channel_buffer: 4096,
        }
    }
}

impl Config {
    /// Full WebSocket URL for the configured symbol's aggregate trade stream.
    pub fn stream_url(&self) -> String {
        format!("{}/stream?streams={}@aggTrade/{}@depth20@100ms", self.ws_url.replace("/ws", ""), self.symbol, self.symbol)
    }
}
