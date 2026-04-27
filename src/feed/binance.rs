//! Binance aggregate trade WebSocket feed.
//!
//! Connects to `wss://stream.binance.com:9443/ws/<symbol>@aggTrade`,
//! deserializes each frame, and pushes [`MarketTick`] into the provided
//! MPSC sender. Implements automatic reconnection with exponential backoff.

use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tracing::{error, info, warn};

use crate::config::Config;
use crate::types::MarketTick;

/// Raw JSON shape from the Binance `@aggTrade` stream.
/// We deserialize into this private struct and then map to our canonical
/// [`MarketTick`], keeping the Binance wire format isolated.
#[derive(Debug, Deserialize)]
struct RawAggTrade {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    qty: String,
    #[serde(rename = "T")]
    timestamp: u64,
    #[serde(rename = "m")]
    is_buyer_maker: bool,
}

impl TryFrom<RawAggTrade> for MarketTick {
    type Error = rust_decimal::Error;

    fn try_from(raw: RawAggTrade) -> std::result::Result<Self, Self::Error> {
        Ok(MarketTick {
            symbol: raw.symbol,
            price: raw.price.parse()?,
            qty: raw.qty.parse()?,
            timestamp: raw.timestamp,
            is_buyer_maker: raw.is_buyer_maker,
        })
    }
}

/// Spawn the WebSocket listener as a long-lived tokio task.
///
/// This task owns the read-half of the WS connection. It never touches
/// strategy logic or account state — it only serializes data and pushes
/// it through the channel, enforcing the "feed never blocks strategy" rule.
pub async fn run_feed(config: Config, tick_tx: mpsc::Sender<MarketTick>) {
    let url = config.stream_url();
    let mut backoff_secs = 1u64;
    const MAX_BACKOFF: u64 = 60;

    loop {
        info!(url = %url, "Connecting to Binance WebSocket…");

        match connect_async(&url).await {
            Ok((ws_stream, _response)) => {
                info!("WebSocket connected.");
                backoff_secs = 1; // reset on success

                let (_write, mut read) = ws_stream.split();

                while let Some(msg_result) = read.next().await {
                    match msg_result {
                        Ok(msg) => {
                            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                                match serde_json::from_str::<RawAggTrade>(&text) {
                                    Ok(raw) => match MarketTick::try_from(raw) {
                                        Ok(tick) => {
                                            if tick_tx.send(tick).await.is_err() {
                                                error!("Tick channel closed — shutting down feed.");
                                                return;
                                            }
                                        }
                                        Err(e) => {
                                            warn!(error = %e, "Decimal parse error, skipping frame");
                                        }
                                    },
                                    Err(e) => {
                                        warn!(error = %e, "JSON deserialization failed, skipping frame");
                                    }
                                }
                            }
                            // Silently ignore Ping/Pong/Binary frames
                        }
                        Err(e) => {
                            error!(error = %e, "WebSocket read error — will reconnect.");
                            break;
                        }
                    }
                }

                warn!("WebSocket stream ended.");
            }
            Err(e) => {
                error!(error = %e, backoff = backoff_secs, "Connection failed.");
            }
        }

        // Exponential backoff before reconnecting
        info!(delay_secs = backoff_secs, "Reconnecting after delay…");
        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF);
    }
}
