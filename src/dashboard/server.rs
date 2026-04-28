//! Axum-based dashboard server with WebSocket broadcast and manual command API.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use crate::types::{DashboardCommand, PortfolioSnapshot};

const DASHBOARD_HTML: &str = include_str!("page.html");
const LOGO_BYTES: &[u8] = include_bytes!("logo.jpg");

async fn serve_logo() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/jpeg")],
        LOGO_BYTES,
    )
}

/// Shared state for the axum routes.
#[derive(Clone)]
struct AppState {
    broadcast_tx: Arc<broadcast::Sender<String>>,
    command_tx: mpsc::Sender<DashboardCommand>,
}

#[derive(Deserialize)]
struct BuyRequest {
    #[serde(default = "default_buy_notional")]
    notional: f64,
}

fn default_buy_notional() -> f64 { 50.0 }

/// Run the dashboard HTTP + WS server.
///
/// Consumes snapshots from the strategy via MPSC, then broadcasts
/// them to all connected WebSocket clients. Also accepts manual
/// commands via POST endpoints.
pub async fn run_dashboard(
    mut snapshot_rx: mpsc::Receiver<PortfolioSnapshot>,
    command_tx: mpsc::Sender<DashboardCommand>,
    port: u16,
) {
    // Broadcast channel: allows multiple WS clients to subscribe
    let (broadcast_tx, _) = broadcast::channel::<String>(256);
    let broadcast_tx = Arc::new(broadcast_tx);

    // Relay task: MPSC → broadcast
    let relay_tx = broadcast_tx.clone();
    tokio::spawn(async move {
        while let Some(snapshot) = snapshot_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&snapshot) {
                let _ = relay_tx.send(json);
            }
        }
    });

    let state = AppState {
        broadcast_tx: broadcast_tx.clone(),
        command_tx,
    };

    let app = Router::new()
        .route("/", get(|| async { Html(DASHBOARD_HTML) }))
        .route("/logo.jpg", get(serve_logo))
        .route("/ws", get(ws_handler))
        .route("/api/buy", post(handle_buy))
        .route("/api/sell-all", post(handle_sell_all))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!(url = format!("http://localhost:{port}"), "Dashboard server starting");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state.broadcast_tx))
}

async fn handle_buy(
    State(state): State<AppState>,
    Json(body): Json<BuyRequest>,
) -> impl IntoResponse {
    let notional = Decimal::try_from(body.notional).unwrap_or(dec!(50));
    if notional <= dec!(0) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "notional must be > 0"})));
    }
    info!(notional = %notional, "Manual buy command received");
    match state.command_tx.send(DashboardCommand::ManualBuy { notional }).await {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({"status": "ok", "action": "buy", "notional": body.notional}))),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "strategy channel closed"}))),
    }
}

async fn handle_sell_all(
    State(state): State<AppState>,
) -> impl IntoResponse {
    info!("PANIC SELL command received");
    match state.command_tx.send(DashboardCommand::PanicSell).await {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({"status": "ok", "action": "sell_all"}))),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "strategy channel closed"}))),
    }
}

async fn handle_ws(mut socket: WebSocket, broadcast_tx: Arc<broadcast::Sender<String>>) {
    let mut rx = broadcast_tx.subscribe();
    info!("Dashboard client connected");

    loop {
        tokio::select! {
            result = rx.recv() => {
                match result {
                    Ok(json) => {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "Dashboard client lagging");
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    info!("Dashboard client disconnected");
}
