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
    /// Print Vulkan GPU information
    VulkanInfo,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { model } => {
            match vulkanize_gguf::parse_gguf_full_from_path(&model) {
                Ok((header, metadata, tensors)) => {
                    match metadata.extract_model_arch() {
                        Ok(arch) => {
                            println!("Model:");
                            if let Some(ref name) = arch.name {
                                println!("  name:         {}", name);
                            }
                            println!("  architecture: {}", arch.architecture);
                            if let Some(ref tokenizer) = arch.tokenizer_model {
                                println!("  tokenizer:    {}", tokenizer);
                            }
                            println!("  blocks:       {}", arch.block_count);
                            println!("  context:      {}", arch.context_length);
                            println!("  embedding:    {}", arch.embedding_length);
                            println!("  ffn:          {}", arch.feed_forward_length);
                            println!("  heads:        {}", arch.attention_head_count);
                            println!("  kv heads:     {}", arch.attention_head_count_kv.unwrap_or(arch.attention_head_count));
                            if let Some(freq) = arch.rope_freq_base {
                                println!("  rope base:    {}", freq);
                            }
                            if let Some(ft) = arch.file_type {
                                println!("  file type:    {}", ft);
                            }
                            println!();
                        }
                        Err(e) => {
                            eprintln!("warning: could not extract architecture: {}", e);
                        }
                    }
                    println!("GGUF Header:");
                    println!("  version: {}", header.version);
                    println!("  tensors: {}", header.tensor_count);
                    println!("  metadata entries: {}", metadata.len());
                    println!();
                    if metadata.is_empty() {
                        println!("(no metadata)");
                    } else {
                        println!("Metadata:");
                        for entry in &metadata.entries {
                            println!("  {} = {}", entry.key, entry.value);
                        }
                    }
                    println!();
                    if tensors.is_empty() {
                        println!("(no tensors)");
                    } else {
                        println!("Tensors ({}):", tensors.len());
                        for t in &tensors.descriptors {
                            let shape = t
                                .shape
                                .iter()
                                .rev()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join("x");
                            println!(
                                "  {:<45} [{:>10}] {} @ offset {}",
                                t.name, shape, t.dtype, t.offset
                            );
                        }
                    }
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
        Commands::VulkanInfo => {
            match vulkanize_vulkan_backend::VulkanContext::new() {
                Ok(ctx) => {
                    let info = ctx.physical_device_info();
                    let queues = ctx.queues();
                    let all_devices = ctx.enumerate_physical_devices().unwrap_or_default();

                    println!("Vulkan Info");
                    println!("===========");
                    println!();

                    // Selected device
                    println!("Selected GPU:");
                    println!("  name:           {}", info.name);
                    println!("  type:           {}", vulkanize_vulkan_backend::format_device_type(info.device_type));
                    println!("  vendor:         {} (0x{:04X})", vulkanize_vulkan_backend::format_vendor_id(info.vendor_id), info.vendor_id);
                    println!("  device id:      0x{:04X}", info.device_id);
                    println!("  api version:    {}", vulkanize_vulkan_backend::format_version(info.api_version));
                    println!("  driver version: {}", info.driver_version);
                    println!("  driver name:    {}", info.driver_name);
                    println!();

                    // Queue families
                    println!("Queue families ({}):", info.queue_families.len());
                    for qf in &info.queue_families {
                        let selected = if qf.index == queues.queue_family_index {
                            " *"
                        } else {
                            ""
                        };
                        println!(
                            "  [{}] {} ({} queues){}",
                            qf.index,
                            vulkanize_vulkan_backend::format_queue_flags(qf.queue_flags),
                            qf.queue_count,
                            selected
                        );
                    }
                    println!("  (* = selected compute queue family)");
                    println!();

                    // Compute queue details
                    println!("Compute queue:");
                    println!("  family: {}", queues.queue_family_index);
                    println!("  index:  {}", queues.queue_index);
                    println!();

                    // Extensions
                    println!("Device extensions ({}):", info.extensions.len());
                    for ext in &info.extensions {
                        println!("  {}", ext);
                    }
                    println!();

                    // All devices
                    if all_devices.len() > 1 {
                        println!("All physical devices ({}):", all_devices.len());
                        for (i, dev) in all_devices.iter().enumerate() {
                            let marker = if dev.name == info.name {
                                " [selected]"
                            } else {
                                ""
                            };
                            println!(
                                "  [{}] {} {} ({}){}",
                                i,
                                dev.name,
                                vulkanize_vulkan_backend::format_vendor_id(dev.vendor_id),
                                vulkanize_vulkan_backend::format_version(dev.api_version),
                                marker
                            );
                        }
                    }

                    // Cleanup
                    if let Err(e) = ctx.wait_idle() {
                        eprintln!("warning: wait_idle failed: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}
