use hivemind_core::{MicroToken, Result, HivemindError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// In-memory token wallet. Thread-safe via atomics.
///
/// TODO: persist balance to `~/.hivemind/wallet.json` between sessions.
#[derive(Debug, Clone)]
pub struct Wallet {
    balance: Arc<AtomicU64>,
}

impl Wallet {
    pub fn new(initial: MicroToken) -> Self {
        Self { balance: Arc::new(AtomicU64::new(initial.0)) }
    }

    pub fn balance(&self) -> MicroToken {
        MicroToken(self.balance.load(Ordering::Relaxed))
    }

    /// Adds earned tokens to the balance.
    pub fn earn(&self, amount: MicroToken) {
        self.balance.fetch_add(amount.0, Ordering::Relaxed);
        tracing::debug!(amount = amount.0, "tokens earned");
    }

    /// Deducts tokens for inference usage.
    ///
    /// Returns an error if the balance would go negative.
    pub fn spend(&self, amount: MicroToken) -> Result<()> {
        let current = self.balance.load(Ordering::Relaxed);
        if current < amount.0 {
            return Err(HivemindError::Ledger(format!(
                "insufficient balance: have {}, need {}",
                MicroToken(current),
                amount
            )));
        }
        self.balance.fetch_sub(amount.0, Ordering::Relaxed);
        tracing::debug!(amount = amount.0, "tokens spent");
        Ok(())
    }

    /// Attempts to spend, returning false instead of an error if balance is low.
    pub fn try_spend(&self, amount: MicroToken) -> bool {
        self.spend(amount).is_ok()
    }
}

impl Default for Wallet {
    fn default() -> Self {
        Self::new(MicroToken::ZERO)
    }
}
