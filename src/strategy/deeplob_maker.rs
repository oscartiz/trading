use crate::{
    strategy::ml::DeepLOB,
    strategy::traits::ExecutionClient,
    types::{ExecutionReport, FillStatus, MarketEvent, OrderRequest, PortfolioSnapshot, Side},
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::{info, warn, error};

#[derive(Deserialize, Debug)]
struct ZScoreParams {
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// Runs the DeepLOB Maker Strategy event loop.
pub async fn run_deeplob_strategy<E: ExecutionClient>(
    client: E,
    mut tick_rx: Receiver<MarketEvent>,
    mut report_rx: Receiver<ExecutionReport>,
    initial_balance: Decimal,
    snapshot_tx: Option<Sender<PortfolioSnapshot>>,
) {
    info!("DeepLOB Maker strategy initializing ML engine...");

    // ── Load ML Weights & Config ──────────────────────────────────────────────
    let device = Device::Cpu; 
    
    let model_path = Path::new("ml/checkpoints/deeplob.safetensors");
    let zscore_path = Path::new("ml/checkpoints/zscore_params.json");
    
    let zscore_params: Option<ZScoreParams> = if zscore_path.exists() {
        match fs::read_to_string(zscore_path) {
            Ok(json_str) => match serde_json::from_str(&json_str) {
                Ok(params) => Some(params),
                Err(e) => { error!("Failed to parse zscore_params.json: {}", e); None }
            },
            Err(e) => { error!("Failed to read zscore_params.json: {}", e); None }
        }
    } else {
        warn!("zscore_params.json not found! Cannot normalize ticks.");
        None
    };

    let model: Option<DeepLOB> = if model_path.exists() {
        match unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device) } {
            Ok(vb) => match DeepLOB::load(vb) {
                Ok(m) => Some(m),
                Err(e) => { error!("Failed to load DeepLOB model architecture: {}", e); None }
            },
            Err(e) => { error!("Failed to map safetensors: {}", e); None }
        }
    } else {
        warn!("deeplob.safetensors not found! Trading is disabled.");
        None
    };

    if model.is_some() && zscore_params.is_some() {
        info!("DeepLOB model and Z-score parameters successfully loaded into Rust.");
    }

    // ── Strategy State ─────────────────────────────────────────────────────────
    let mut quote_balance = initial_balance;
    let mut crypto_qty = Decimal::ZERO;
    let mut last_price = Decimal::ZERO;
    let mut order_id_counter = 0;
    
    // Sliding Window
    let mut orderbook_buffer: VecDeque<[f32; 40]> = VecDeque::with_capacity(100);
    
    // Hysteresis State
    let mut position_entry_price = Decimal::ZERO;
    let mut ticks_held = 0;
    
    let mut current_prediction = "Awaiting 100 ticks...".to_string();
    let mut last_log_time = std::time::Instant::now();

    loop {
        tokio::select! {
            Some(tick) = tick_rx.recv() => {
                match tick {
                    MarketEvent::Tick(t) => {
                        last_price = t.price;
                    }
                    MarketEvent::Depth(d) => {
                        let best_bid = d.bids.first().map(|l| l.price).unwrap_or(last_price);
                        let best_ask = d.asks.first().map(|l| l.price).unwrap_or(last_price);
                        
                        if best_bid.is_zero() || best_ask.is_zero() {
                            continue;
                        }
                        
                        // Extract features
                        let mut feature_row = [0.0f32; 40];
                        for i in 0..10 {
                            let ask = d.asks.get(i);
                            let bid = d.bids.get(i);
                            
                            feature_row[i * 4 + 0] = ask.map(|a| a.price.to_f32().unwrap_or(0.0)).unwrap_or(0.0);
                            feature_row[i * 4 + 1] = ask.map(|a| a.qty.to_f32().unwrap_or(0.0)).unwrap_or(0.0);
                            feature_row[i * 4 + 2] = bid.map(|b| b.price.to_f32().unwrap_or(0.0)).unwrap_or(0.0);
                            feature_row[i * 4 + 3] = bid.map(|b| b.qty.to_f32().unwrap_or(0.0)).unwrap_or(0.0);
                        }
                        
                        // Normalize and buffer
                        if let Some(z) = &zscore_params {
                            for i in 0..40 {
                                feature_row[i] = (feature_row[i] - z.mean[i]) / z.std[i];
                            }
                            
                            if orderbook_buffer.len() == 100 {
                                orderbook_buffer.pop_front();
                            }
                            orderbook_buffer.push_back(feature_row);
                            
                            let len = orderbook_buffer.len();
                            if len % 10 == 0 && len < 100 {
                                info!("DeepLOB buffer filling: {}/100 ticks", len);
                            }
                        }
                        
                        if crypto_qty > Decimal::ZERO {
                            ticks_held += 1;
                        }

                        // Inference
                        if orderbook_buffer.len() == 100 {
                            if let Some(m) = &model {
                                let mut flat = Vec::with_capacity(4000);
                                for row in &orderbook_buffer {
                                    flat.extend_from_slice(row);
                                }
                                
                                match Tensor::from_vec(flat, (1, 1, 100, 40), &device) {
                                    Err(e) => error!("Tensor creation failed: {}", e),
                                    Ok(input) => {
                                        match m.predict(&input) {
                                            Err(e) => error!("Inference failed: {}", e),
                                            Ok(probs) => {
                                                let prob_down = probs[0];
                                                let prob_stat = probs[1];
                                                let prob_up = probs[2];
                                                
                                                current_prediction = format!(
                                                    "Up: {:.1}%, Down: {:.1}%, Stat: {:.1}%", 
                                                    prob_up * 100.0, prob_down * 100.0, prob_stat * 100.0
                                                );
                                                
                                                // Periodic logging to show activity
                                                if last_log_time.elapsed().as_secs() >= 10 {
                                                    info!("Current Prediction: {}", current_prediction);
                                                    last_log_time = std::time::Instant::now();
                                                }
                                                
                                                // Maker Threshold Execution Logic
                                                let prob_threshold = 0.85;
                                                let stat_threshold = 0.10;
                                                
                                                let mut unrealized_profit = Decimal::ZERO;
                                                if crypto_qty > Decimal::ZERO && position_entry_price > Decimal::ZERO {
                                                    unrealized_profit = (best_bid - position_entry_price) / position_entry_price;
                                                }
                                                let can_exit = ticks_held >= 20 || unrealized_profit > rust_decimal_macros::dec!(0.001);
                                                
                                                if prob_up > prob_threshold && prob_stat < stat_threshold && quote_balance > Decimal::ZERO {
                                                    // Place Maker Bid
                                                    order_id_counter += 1;
                                                    let qty = (quote_balance * rust_decimal_macros::dec!(0.99)) / best_bid;
                                                    let rounded_qty = qty.round_dp(8);
                                                    
                                                    if rounded_qty > Decimal::ZERO {
                                                        let order = OrderRequest {
                                                            id: order_id_counter,
                                                            symbol: "BTCUSDT".to_string(),
                                                            side: Side::Buy,
                                                            qty: rounded_qty,
                                                            price: best_bid,
                                                        };
                                                        info!(id = order.id, price = %best_bid, conf = %prob_up, "DeepLOB Maker BUY Signal");
                                                        let _ = client.submit_order(order).await;
                                                    }
                                                } else if prob_down > prob_threshold && prob_stat < stat_threshold && crypto_qty > Decimal::ZERO && can_exit {
                                                    // Place Maker Ask
                                                    order_id_counter += 1;
                                                    let order = OrderRequest {
                                                        id: order_id_counter,
                                                        symbol: "BTCUSDT".to_string(),
                                                        side: Side::Sell,
                                                        qty: crypto_qty,
                                                        price: best_ask,
                                                    };
                                                    info!(id = order.id, price = %best_ask, conf = %prob_down, "DeepLOB Maker SELL Signal");
                                                    let _ = client.submit_order(order).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Some(report) = report_rx.recv() => {
                info!(
                    order_id = %report.order_id,
                    side = ?report.side,
                    status = ?report.status,
                    filled = %report.filled_qty,
                    "Execution report received"
                );
                
                if report.status == FillStatus::Filled {
                    if report.side == Side::Buy {
                        crypto_qty += report.filled_qty;
                        quote_balance -= report.filled_qty * report.fill_price;
                        
                        position_entry_price = report.fill_price;
                        ticks_held = 0;
                    } else {
                        crypto_qty -= report.filled_qty;
                        quote_balance += report.filled_qty * report.fill_price;
                        
                        position_entry_price = Decimal::ZERO;
                        ticks_held = 0;
                    }
                }
            }

            else => {
                info!("All channels closed, shutting down DeepLOB Maker.");
                break;
            }
        }

        if let Some(tx) = &snapshot_tx {
            let portfolio_value = quote_balance + (crypto_qty * last_price);
            let allocation_pct = if portfolio_value > Decimal::ZERO {
                (crypto_qty * last_price) / portfolio_value * rust_decimal_macros::dec!(100)
            } else {
                Decimal::ZERO
            };
            
            let mut unrealized_pnl = Decimal::ZERO;
            if crypto_qty > Decimal::ZERO && position_entry_price > Decimal::ZERO {
                unrealized_pnl = (last_price - position_entry_price) * crypto_qty;
            }
            
            let snapshot = PortfolioSnapshot {
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                price: last_price,
                quote_balance,
                crypto_qty,
                portfolio_value,
                allocation_pct,
                cost_basis: position_entry_price,
                unrealized_pnl,
                rsi: None,
                ml_prediction: Some(current_prediction.clone()),
                event_history: vec![],
            };
            let _ = tx.try_send(snapshot);
        }
    }
}
