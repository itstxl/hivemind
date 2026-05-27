use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::{io, time::Duration};

#[derive(Debug, Clone, PartialEq, Eq)]
enum AppState {
    Connecting,
    Ready,
    Thinking,
}

#[derive(Debug, Clone)]
struct Message {
    role: &'static str,
    content: String,
}

struct App {
    messages: Vec<Message>,
    input: String,
    state: AppState,
    model_name: String,
    scroll_offset: usize,
}

impl App {
    fn new(model_name: String) -> Self {
        Self {
            messages: vec![Message {
                role: "system",
                content: "Connecting to the Hivemind network...".into(),
            }],
            input: String::new(),
            state: AppState::Connecting,
            model_name,
            scroll_offset: 0,
        }
    }

    fn on_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Enter => {
                self.submit_message();
            }
            KeyCode::Char(c) => {
                if self.state == AppState::Ready {
                    self.input.push(c);
                }
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Up => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
            }
            KeyCode::Down => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            _ => {}
        }
        false
    }

    fn submit_message(&mut self) {
        if self.state != AppState::Ready {
            return;
        }
        let text = std::mem::take(&mut self.input).trim().to_string();
        if text.is_empty() {
            return;
        }
        self.messages.push(Message { role: "you", content: text });
        self.state = AppState::Thinking;

        // TODO: send to network pipeline; for now push a stub reply
        self.messages.push(Message {
            role: "assistant",
            content: "Network inference is not yet wired up. \
                      The pipeline assembler and shard servers are being built — \
                      check `hivemind status` for network readiness."
                .into(),
        });
        self.state = AppState::Ready;
        self.scroll_offset = 0;
    }

    fn simulate_connect(&mut self) {
        if self.state == AppState::Connecting {
            self.state = AppState::Ready;
            self.messages.push(Message {
                role: "system",
                content: format!(
                    "Connected (mock). Model: {}  |  Type a message and press Enter.",
                    self.model_name
                ),
            });
        }
    }
}

/// Runs the chat TUI. Must be called from a `spawn_blocking` context.
pub fn run_blocking(model_name: &str, _temperature: f32) -> Result<()> {
    // Install a panic hook that restores the terminal before unwinding
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Safety: we're already panicking; best-effort cleanup
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        default_panic(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(&mut terminal, model_name);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    model_name: &str,
) -> Result<()> {
    let mut app = App::new(model_name.to_string());
    let mut tick_count: u32 = 0;

    loop {
        terminal.draw(|frame| render(frame, &app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Ctrl-C or Ctrl-Q exits
                if matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q'))
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                {
                    return Ok(());
                }
                // Esc also exits
                if key.code == KeyCode::Esc {
                    return Ok(());
                }
                app.on_key(key.code);
            }
        }

        // Simulate network connection after ~500 ms
        tick_count += 1;
        if tick_count == 5 {
            app.simulate_connect();
        }
    }
}

fn render(frame: &mut ratatui::Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    // Header title
    let title = format!(
        " Hivemind Chat  |  {}  |  {} ",
        app.model_name,
        match app.state {
            AppState::Connecting => "connecting...",
            AppState::Ready      => "ready",
            AppState::Thinking   => "thinking...",
        }
    );

    // Message list
    let items: Vec<ListItem> = app
        .messages
        .iter()
        .map(|m| {
            let (role_style, prefix) = match m.role {
                "you"       => (Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD), "You"),
                "assistant" => (Style::default().fg(Color::Green).add_modifier(Modifier::BOLD), "AI"),
                _           => (Style::default().fg(Color::DarkGray), "•"),
            };
            let line = Line::from(vec![
                Span::styled(format!(" {prefix}: "), role_style),
                Span::raw(&m.content),
            ]);
            ListItem::new(line)
        })
        .collect();

    let messages = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(messages, chunks[0]);

    // Input box
    let input_style = if app.state == AppState::Ready {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let input_display = if app.state == AppState::Thinking {
        " generating...".to_string()
    } else {
        format!(" {}_", app.input)
    };
    let input_widget = Paragraph::new(input_display)
        .style(input_style)
        .block(Block::default().borders(Borders::ALL).title(" Message "));
    frame.render_widget(input_widget, chunks[1]);

    // Status bar
    let help = Paragraph::new("  Ctrl-C / Esc: quit   Enter: send   ↑↓: scroll")
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}
