#![allow(dead_code)] // Stub — scheduler will be called from the real gRPC handlers

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tracing::debug;

/// Controls how aggressively the daemon serves pipeline requests
/// based on local user activity and configured resource limits.
pub struct ResourceScheduler {
    /// True when the local user has an active session (chat/complete).
    user_active: Arc<AtomicBool>,
    /// Current number of in-flight pipeline requests being served.
    active_pipelines: Arc<AtomicU32>,
    /// Hard cap from config.
    max_concurrent_pipelines: u32,
}

impl ResourceScheduler {
    pub fn new(max_concurrent_pipelines: u32) -> Self {
        Self {
            user_active: Arc::new(AtomicBool::new(false)),
            active_pipelines: Arc::new(AtomicU32::new(0)),
            max_concurrent_pipelines,
        }
    }

    /// Called when the local user starts a session; reduces serving aggressiveness.
    pub fn set_user_active(&self, active: bool) {
        self.user_active.store(active, Ordering::Relaxed);
        debug!(active, "user activity state changed");
    }

    /// Returns true if the daemon should accept a new pipeline request.
    pub fn can_accept_pipeline(&self) -> bool {
        let current = self.active_pipelines.load(Ordering::Relaxed);
        let limit = if self.user_active.load(Ordering::Relaxed) {
            // Throttle to half capacity when the user is active
            self.max_concurrent_pipelines / 2
        } else {
            self.max_concurrent_pipelines
        };
        current < limit
    }

    /// Acquires a pipeline slot. Returns a guard that releases the slot on drop.
    pub fn acquire_pipeline(&self) -> Option<PipelineGuard> {
        if !self.can_accept_pipeline() {
            return None;
        }
        self.active_pipelines.fetch_add(1, Ordering::Relaxed);
        Some(PipelineGuard { counter: Arc::clone(&self.active_pipelines) })
    }

    pub fn active_pipeline_count(&self) -> u32 {
        self.active_pipelines.load(Ordering::Relaxed)
    }
}

/// RAII guard that decrements the active pipeline counter on drop.
pub struct PipelineGuard {
    counter: Arc<AtomicU32>,
}

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}
