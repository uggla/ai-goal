// src/lib.rs
use anyhow::{Context, Result, bail};
use futures::future::join_all;
use reqwest::Client;
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;

// Define a struct for model information
#[derive(Debug, Clone)]
struct ModelInfo {
    name: &'static str,
    url: &'static str,
}

// Define model URLs and expected filenames using the struct
const MODEL_FILES: [ModelInfo; 2] = [
    ModelInfo {
        name: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    },
    ModelInfo {
        name: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    },
    // Add other models here if needed
];

const MODELS_DIR: &str = "models";
const OLLAMA_API_URL: &str = "http://localhost:11434/api/tags";

// Define a list of target Ollama models
const OLLAMA_TARGET_MODELS: &[&str] = &["mistral", "llama3", "gemma"];

/// Checks if ffmpeg is installed and accessible.
fn check_ffmpeg_availability() -> Result<()> {
    println!("1. Checking for ffmpeg...");
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("Failed to execute ffmpeg command. Is it installed and in PATH?")?;

    if output.status.success() {
        println!("   ffmpeg found.");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ffmpeg seems to be installed but '-version' command failed: {}",
            stderr
        )
    }
}

/// Checks for Whisper model files and downloads them in parallel if missing.
async fn check_and_download_models(http_client: &Client) -> Result<()> {
    println!("\n2. Checking Whisper models...");
    let models_path_obj = Path::new(MODELS_DIR);

    if !models_path_obj.exists() {
        println!("   '{}' directory does not exist, creating...", MODELS_DIR);
        tokio::fs::create_dir_all(models_path_obj)
            .await
            .with_context(|| format!("Could not create directory '{}'", MODELS_DIR))?;
    }

    let mut download_futures = Vec::new();

    for model_info in MODEL_FILES.iter() {
        let model_path = models_path_obj.join(model_info.name);
        if model_path.exists() {
            println!("   Model '{}' found.", model_info.name);
        } else {
            println!(
                "   Model '{}' not found. Starting download from {}...",
                model_info.name, model_info.url
            );
            let client_clone = http_client.clone();
            let model_info_clone = model_info.clone();
            let model_path_clone = model_path.clone();

            download_futures.push(tokio::spawn(async move {
                let response = client_clone
                    .get(model_info_clone.url)
                    .send()
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to send download request for model '{}' from {}",
                            model_info_clone.name, model_info_clone.url
                        )
                    })?;

                if response.status().is_success() {
                    let mut dest_file =
                        TokioFile::create(&model_path_clone)
                            .await
                            .with_context(|| {
                                format!(
                                    "Could not create file for model '{}' at {:?}",
                                    model_info_clone.name, model_path_clone
                                )
                            })?;

                    let mut stream = response.bytes_stream();
                    while let Some(item) = futures::StreamExt::next(&mut stream).await {
                        let chunk = item.with_context(|| {
                            format!("Error while downloading model '{}'", model_info_clone.name)
                        })?;
                        dest_file.write_all(&chunk).await.with_context(|| {
                            format!(
                                "Error writing chunk for model '{}' to {:?}",
                                model_info_clone.name, model_path_clone
                            )
                        })?;
                    }
                    dest_file.flush().await.with_context(|| {
                        format!("Error flushing model file '{}'", model_info_clone.name)
                    })?;
                    println!(
                        "   Model '{}' downloaded successfully.",
                        model_info_clone.name
                    );
                    Ok(())
                } else {
                    bail!(
                        "Failed to download model '{}'. Status: {}",
                        model_info_clone.name,
                        response.status()
                    );
                }
            }));
        }
    }

    let results = join_all(download_futures).await;
    for result in results {
        match result {
            Ok(inner_result) => inner_result?,
            Err(join_error) => bail!("Download task failed: {}", join_error),
        }
    }

    Ok(())
}

/// Checks if the Ollama API is responsive and if any of the specified target models are available.
async fn check_ollama_api_and_model(http_client: &Client) -> Result<()> {
    println!(
        "\n3. Checking Ollama API and target models: {:?}...",
        OLLAMA_TARGET_MODELS
    );

    let response = http_client
        .get(OLLAMA_API_URL)
        .send()
        .await
        .with_context(|| {
            format!(
                "Could not contact Ollama API at {}. Ensure Ollama is running and accessible.",
                OLLAMA_API_URL
            )
        })?;

    if response.status().is_success() {
        println!("   Ollama API is responsive.");
        let data: Value = response
            .json()
            .await
            .context("Could not parse JSON response from Ollama API")?;

        let ollama_models_on_system: Vec<String> = data
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Malformed Ollama API response: 'models' field missing or not an array."
                )
            })?
            .iter()
            .filter_map(|m| m.get("name").and_then(Value::as_str).map(String::from))
            .collect();

        let mut found_target_models_on_system = Vec::new();
        for target_model_name in OLLAMA_TARGET_MODELS.iter() {
            if ollama_models_on_system
                .iter()
                .any(|sys_model_name| sys_model_name.starts_with(target_model_name))
            {
                found_target_models_on_system.push(*target_model_name);
            }
        }

        if !found_target_models_on_system.is_empty() {
            println!(
                "   Found target Ollama model(s) on system: {:?}.",
                found_target_models_on_system
            );
            Ok(())
        } else {
            bail!(
                "None of the target Ollama models ({:?}) were found. Available models on system: {:?}",
                OLLAMA_TARGET_MODELS,
                ollama_models_on_system
            )
        }
    } else {
        bail!(
            "Ollama API responded with an error status: {}. Ensure Ollama is running correctly.",
            response.status()
        )
    }
}

/// Main public function to orchestrate all system requirement checks.
/// This function is made public so it can be called from main.rs.
pub async fn check_all_system_requirements() -> Result<()> {
    println!("Checking system prerequisites...");

    check_ffmpeg_availability()?;

    let http_client = Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("Could not build HTTP client")?;

    check_and_download_models(&http_client).await?;
    check_ollama_api_and_model(&http_client).await?;

    println!("\nAll prerequisites are met!");
    Ok(())
}
