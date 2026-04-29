//! Binance multiplexed WebSocket feed.
//!
//! Connects to combined streams for `aggTrade` and `depth20@100ms`,
//! deserializes each frame, and pushes [`MarketEvent`] into the provided
//! MPSC sender. Implements automatic reconnection with exponential backoff.

use futures_util::StreamExt;

use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tracing::{error, info, warn};

use crate::config::Config;
use crate::types::{DepthLevel, DepthSnapshot, MarketEvent, MarketTick};

/// Container for multiplexed streams
#[derive(Debug, Deserialize)]
struct MultiplexedMessage {
    pub stream: String,
    pub data: serde_json::Value,
}

/// Raw JSON shape from the Binance `@aggTrade` stream.
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

/// Raw JSON shape from the Binance `@depth20` stream.
#[derive(Debug, Deserialize)]
struct RawDepth {
    #[serde(default)]
    lastUpdateId: u64,
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
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

impl TryFrom<RawDepth> for DepthSnapshot {
    type Error = rust_decimal::Error;

    fn try_from(raw: RawDepth) -> std::result::Result<Self, Self::Error> {
        let mut bids = Vec::with_capacity(raw.bids.len());
        for b in raw.bids {
            bids.push(DepthLevel {
                price: b[0].parse()?,
                qty: b[1].parse()?,
            });
        }
        
        let mut asks = Vec::with_capacity(raw.asks.len());
        for a in raw.asks {
            asks.push(DepthLevel {
                price: a[0].parse()?,
                qty: a[1].parse()?,
            });
        }
        
        // Approximate timestamp using system time since depth20 doesn't provide it
        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;

        Ok(DepthSnapshot {
            timestamp,
            bids,
            asks,
        })
    }
}

pub async fn run_feed(config: Config, event_tx: mpsc::Sender<MarketEvent>) {
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
                                match serde_json::from_str::<MultiplexedMessage>(&text) {
                                    Ok(multi) => {
                                        if multi.stream.ends_with("@aggTrade") {
                                            if let Ok(raw) = serde_json::from_value::<RawAggTrade>(multi.data) {
                                                if let Ok(tick) = MarketTick::try_from(raw) {
                                                    if event_tx.send(MarketEvent::Tick(tick)).await.is_err() {
                                                        error!("Event channel closed — shutting down feed.");
                                                        return;
                                                    }
                                                }
                                            }
                                        } else if multi.stream.ends_with("@depth20@100ms") {
                                            if let Ok(raw) = serde_json::from_value::<RawDepth>(multi.data) {
                                                if let Ok(depth) = DepthSnapshot::try_from(raw) {
                                                    if event_tx.send(MarketEvent::Depth(depth)).await.is_err() {
                                                        error!("Event channel closed — shutting down feed.");
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        // Ignore parsing error for top-level object
                                    }
                                }
                            }
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
