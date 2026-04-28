//! Axum-based dashboard server with WebSocket broadcast.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use crate::types::PortfolioSnapshot;

const DASHBOARD_HTML: &str = include_str!("page.html");
const LOGO_BYTES: &[u8] = include_bytes!("logo.jpg");

async fn serve_logo() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/jpeg")],
        LOGO_BYTES,
    )
}

/// Run the dashboard HTTP + WS server.
///
/// Consumes snapshots from the strategy via MPSC, then broadcasts
/// them to all connected WebSocket clients.
pub async fn run_dashboard(
    mut snapshot_rx: mpsc::Receiver<PortfolioSnapshot>,
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

    let app = Router::new()
        .route("/", get(|| async { Html(DASHBOARD_HTML) }))
        .route("/logo.jpg", get(serve_logo))
        .route("/ws", get({
            let tx = broadcast_tx.clone();
            move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| handle_ws(socket, tx))
            }
        }));

    let addr = format!("0.0.0.0:{port}");
    info!(url = format!("http://localhost:{port}"), "Dashboard server starting");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
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
