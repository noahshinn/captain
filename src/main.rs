use clap::{Parser, Subcommand};

pub mod autocomplete;
pub mod embeddings;
pub mod image_analysis;
pub mod llm;
pub mod prompts;
pub mod screenshot;
pub mod search;
pub mod shell;
pub mod trajectory;
pub mod utils;

#[derive(Parser)]
#[command(name = "captain")]
#[command(about = "Captain: helps you with your work")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Shell {},
    Autocomplete {},
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY");
    let openai_key = std::env::var("OPENAI_API_KEY");
    if anthropic_key.is_err() {
        return Err("ANTHROPIC_API_KEY is not set".into());
    }
    if openai_key.is_err() {
        return Err("OPENAI_API_KEY is not set".into());
    }

    let cli = Cli::parse();
    match cli.command {
        Commands::Shell {} => shell::run_shell().await,
        Commands::Autocomplete {} => autocomplete::run_autocomplete().await,
    }
}
