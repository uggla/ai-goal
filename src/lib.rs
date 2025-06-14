// src/lib.rs
use anyhow::{Context, Result, bail};
use futures::future::join_all;
use reqwest::Client;
use serde_json::Value;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Define a struct for model information
#[derive(Debug, Clone)]
struct ModelInfo {
    name: &'static str,
    url: &'static str,
}

// Define model URLs and expected filenames using the struct
const MODEL_FILES: [ModelInfo; 2] = [
    ModelInfo {
        name: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    ModelInfo {
        name: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
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

pub fn convert_to_wav_mono_16k<P: AsRef<Path>>(input: P, output_dir: P) -> Result<PathBuf, String> {
    let input_path = input.as_ref();
    let output_base = output_dir.as_ref().join("audio");

    if !input_path.exists() {
        return Err(format!("Fichier introuvable: {}", input_path.display()));
    }

    // Créer le dossier audio/ si nécessaire
    fs::create_dir_all(&output_base)
        .map_err(|e| format!("Impossible de créer le dossier de sortie: {}", e))?;

    let input_filename = input_path
        .file_stem()
        .ok_or("Nom de fichier invalide")?
        .to_string_lossy();

    let output_path = output_base.join(format!("{}_mono16k.wav", input_filename));

    let status = Command::new("ffmpeg")
        .args([
            "-i",
            input_path.to_str().unwrap(),
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path.to_str().unwrap(),
            "-y",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Erreur lors de l'exécution de ffmpeg: {}", e))?;

    if !status.success() {
        return Err(format!("ffmpeg a échoué avec le statut: {}", status));
    }

    Ok(output_path)
}

pub fn transcribe_audio<P: AsRef<Path>>(
    audio_path: P,
    output_dir: P,
    model_path: P,
    model_name: &str,
    n_threads: i32,
    language: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let model_path = model_path.as_ref();
    let transcript_dir = output_dir
        .as_ref()
        .join(format!("transcript_{}", model_name));
    fs::create_dir_all(&transcript_dir)?;

    whisper_rs::install_logging_hooks();
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset

    context_param.dtw_parameters.mode = match model_name {
        "tiny" => whisper_rs::DtwMode::ModelPreset {
            model_preset: whisper_rs::DtwModelPreset::Tiny,
        },
        "base" => whisper_rs::DtwMode::ModelPreset {
            model_preset: whisper_rs::DtwModelPreset::Base,
        },
        _ => panic!("Model unknown"),
    };

    //     context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
    //     model_preset: whisper_rs::DtwModelPreset::Tiny,
    // };

    // // Enable DTW token level timestamp for unknown model by providing custom aheads
    // // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    // let custom_aheads = [
    //     (3, 1),
    //     (4, 2),
    //     (4, 3),
    //     (4, 7),
    //     (5, 1),
    //     (5, 2),
    //     (5, 4),
    //     (5, 6),
    // ]
    // .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
    //     n_text_layer,
    //     n_head,
    // });
    // context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
    //     aheads: &custom_aheads,
    // };

    let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), context_param)
        .expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(n_threads);
    // Enable translation.
    if language.is_some() {
        params.set_translate(true);
    } else {
        params.set_translate(false);
    }
    params.set_language(language);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Open the audio file.
    let reader = hound::WavReader::open(audio_path).expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.expect("Invalid sample"))
        .collect();
    let mut audio = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    // Convert audio to 16KHz mono f32 samples, as required by the model.
    // These utilities are provided for convenience, but can be replaced with custom conversion logic.
    // SIMD variants of these functions are also available on nightly Rust (see the docs).
    // if channels == 2 {
    // audio = whisper_rs::convert_stereo_to_mono_audio(&audio).expect("Conversion error");
    if channels != 1 {
        panic!(">1 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }

    // Run the model.
    state.full(params, &audio[..]).expect("failed to run model");

    // Create a file to write the transcript to.
    let transcript_path = transcript_dir.join("transcript.txt");
    let mut file = File::create(&transcript_path).expect("failed to create file");

    // Iterate through the segments of the transcript.
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = match state.full_get_segment_text(i) {
            Ok(s) => s,
            Err(_) => "<unreadable utf-8>".to_string(),
        };

        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get end timestamp");

        let first_token_dtw_ts = if let Ok(token_count) = state.full_n_tokens(i) {
            if token_count > 0 {
                if let Ok(token_data) = state.full_get_token_data(i, 0) {
                    token_data.t_dtw
                } else {
                    -1i64
                }
            } else {
                -1i64
            }
        } else {
            -1i64
        };
        // Print the segment to stdout.
        println!(
            "[{} - {} ({})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    Ok(transcript_path)
}
