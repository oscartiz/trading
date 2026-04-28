//! # Real-Time Dashboard
//!
//! Serves a web UI for live portfolio visualization.
//! - `GET /`  → Single-page dashboard with charts
//! - `WS /ws` → Live portfolio snapshot stream

pub mod server;
