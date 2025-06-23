use super::MODEL_FILES;
use super::MODELS_DIR;
use super::OLLAMA_API_URL;
use super::OLLAMA_TARGET_MODELS;
use anyhow::{Context, Result, bail};
use futures::future::join_all;
use reqwest::Client;
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use tracing::info;

/// Checks if ffmpeg is installed and accessible.
pub(crate) fn check_ffmpeg_availability() -> Result<()> {
    info!("1. Checking for ffmpeg...");
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("Failed to execute ffmpeg command. Is it installed and in PATH?")?;

    if output.status.success() {
        info!("   ffmpeg found.");
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
pub(crate) async fn check_and_download_models(http_client: &Client) -> Result<()> {
    info!("2. Checking Whisper models...");
    let models_path_obj = Path::new(MODELS_DIR);

    if !models_path_obj.exists() {
        info!("   '{}' directory does not exist, creating...", MODELS_DIR);
        tokio::fs::create_dir_all(models_path_obj)
            .await
            .with_context(|| format!("Could not create directory '{}'", MODELS_DIR))?;
    }

    let mut download_futures = Vec::new();

    for model_info in MODEL_FILES.iter() {
        let model_path = models_path_obj.join(model_info.filename);
        if model_path.exists() {
            info!("   Model '{}' found.", model_info.filename);
        } else {
            info!(
                "   Model '{}' not found. Starting download from {}...",
                model_info.filename, model_info.url
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
                            model_info_clone.filename, model_info_clone.url
                        )
                    })?;

                if response.status().is_success() {
                    let mut dest_file =
                        TokioFile::create(&model_path_clone)
                            .await
                            .with_context(|| {
                                format!(
                                    "Could not create file for model '{}' at {:?}",
                                    model_info_clone.filename, model_path_clone
                                )
                            })?;

                    let mut stream = response.bytes_stream();
                    while let Some(item) = futures::StreamExt::next(&mut stream).await {
                        let chunk = item.with_context(|| {
                            format!(
                                "Error while downloading model '{}'",
                                model_info_clone.filename
                            )
                        })?;
                        dest_file.write_all(&chunk).await.with_context(|| {
                            format!(
                                "Error writing chunk for model '{}' to {:?}",
                                model_info_clone.filename, model_path_clone
                            )
                        })?;
                    }
                    dest_file.flush().await.with_context(|| {
                        format!("Error flushing model file '{}'", model_info_clone.filename)
                    })?;
                    info!(
                        "   Model '{}' downloaded successfully.",
                        model_info_clone.filename
                    );
                    Ok(())
                } else {
                    bail!(
                        "Failed to download model '{}'. Status: {}",
                        model_info_clone.filename,
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
pub(crate) async fn check_ollama_api_and_model(http_client: &Client) -> Result<()> {
    info!(
        "3. Checking Ollama API and target models: {:?}...",
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
        info!("   Ollama API is responsive.");
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
            info!(
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
