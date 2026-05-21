use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(name = "vulkanize")]
#[command(about = "AMD-first GGUF inference via Vulkan compute")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Dump GGUF model metadata
    Inspect {
        /// Path to the GGUF model file
        model: String,
    },
    /// Run inference with a prompt
    Generate {
        /// Path to the GGUF model file
        model: String,
        /// Prompt to generate from
        #[arg(default_value = "")]
        prompt: String,
    },
    /// Start OpenAI-compatible API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "8000")]
        port: u16,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { model } => {
            match vulkanize_gguf::parse_header_from_path(&model) {
                Ok(header) => {
                    println!("GGUF Header:");
                    println!("  version: {}", header.version);
                    println!("  tensors: {}", header.tensor_count);
                    println!("  metadata entries: {}", header.metadata_kv_count);
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        Commands::Generate { model: _, prompt: _ } => {
            eprintln!("not yet implemented");
            process::exit(1);
        }
        Commands::Serve { host: _, port: _ } => {
            eprintln!("not yet implemented");
            process::exit(1);
        }
    }
}
