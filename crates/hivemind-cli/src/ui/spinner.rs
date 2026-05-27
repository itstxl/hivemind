use std::io::{self, Write};

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A simple terminal spinner for blocking operations.
pub struct Spinner {
    message: String,
    frame: usize,
}

impl Spinner {
    pub fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        let mut s = Self { message: message.clone(), frame: 0 };
        s.render();
        s
    }

    pub fn tick(&mut self) {
        self.frame = (self.frame + 1) % FRAMES.len();
        self.render();
    }

    /// Clears the spinner line and prints a completion message.
    pub fn finish(self, message: impl Into<String>) {
        eprint!("\r\x1b[K  {} {}\n", "", message.into());
        let _ = io::stderr().flush();
    }

    fn render(&self) {
        eprint!("\r  {} {}", FRAMES[self.frame], self.message);
        let _ = io::stderr().flush();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        // Clear the spinner line if not explicitly finished
        eprint!("\r\x1b[K");
        let _ = io::stderr().flush();
    }
}
