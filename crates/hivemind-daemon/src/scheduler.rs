#![allow(dead_code)] // Stub — scheduler will be called from the real gRPC handlers

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

/// Controls how aggressively the daemon serves pipeline requests
/// based on local user activity, drain state, and configured resource limits.
pub struct ResourceScheduler {
    /// True when the local user has an active session (chat/complete).
    user_active: Arc<AtomicBool>,
    /// True once the daemon has begun a graceful drain: no new pipelines are
    /// accepted, and shutdown waits for in-flight ones to finish.
    draining: Arc<AtomicBool>,
    /// Current number of in-flight pipeline requests being served.
    active_pipelines: Arc<AtomicU32>,
    /// Hard cap from config.
    max_concurrent_pipelines: u32,
}

impl ResourceScheduler {
    pub fn new(max_concurrent_pipelines: u32) -> Self {
        Self {
            user_active: Arc::new(AtomicBool::new(false)),
            draining: Arc::new(AtomicBool::new(false)),
            active_pipelines: Arc::new(AtomicU32::new(0)),
            max_concurrent_pipelines,
        }
    }

    /// Called when the local user starts a session; reduces serving aggressiveness.
    pub fn set_user_active(&self, active: bool) {
        self.user_active.store(active, Ordering::Relaxed);
        debug!(active, "user activity state changed");
    }

    /// Begins a graceful drain: stop accepting new pipelines, finish in-flight
    /// ones. Triggered by SIGTERM/ctrl-c today; lid-close, battery-threshold,
    /// and user-GPU-contention signals should route here too.
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Relaxed);
        debug!("drain started — no longer accepting pipelines");
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Relaxed)
    }

    /// True when draining and every in-flight pipeline has completed.
    pub fn is_drained(&self) -> bool {
        self.is_draining() && self.active_pipeline_count() == 0
    }

    /// Waits until all in-flight pipelines finish or `timeout` elapses.
    /// Returns true if the drain completed cleanly.
    pub async fn wait_drained(&self, timeout: Duration) -> bool {
        let poll = Duration::from_millis(50);
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if self.is_drained() {
                return true;
            }
            tokio::time::sleep(poll).await;
        }
        self.is_drained()
    }

    /// Returns true if the daemon should accept a new pipeline request.
    pub fn can_accept_pipeline(&self) -> bool {
        if self.is_draining() {
            return false;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draining_rejects_new_pipelines() {
        let s = ResourceScheduler::new(4);
        let _guard = s.acquire_pipeline().unwrap();
        s.begin_drain();
        assert!(s.acquire_pipeline().is_none());
        assert!(!s.is_drained(), "still one pipeline in flight");
    }

    #[test]
    fn drain_completes_when_guards_drop() {
        let s = ResourceScheduler::new(4);
        let guard = s.acquire_pipeline().unwrap();
        s.begin_drain();
        assert!(!s.is_drained());
        drop(guard);
        assert!(s.is_drained());
    }

    #[tokio::test]
    async fn wait_drained_times_out_with_stuck_pipeline() {
        let s = ResourceScheduler::new(4);
        let _guard = s.acquire_pipeline().unwrap();
        s.begin_drain();
        assert!(!s.wait_drained(Duration::from_millis(120)).await);
    }

    #[tokio::test]
    async fn wait_drained_returns_when_pipelines_finish() {
        let s = Arc::new(ResourceScheduler::new(4));
        let guard = s.acquire_pipeline().unwrap();
        s.begin_drain();
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move { s2.wait_drained(Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(60)).await;
        drop(guard);
        assert!(waiter.await.unwrap());
    }
}
