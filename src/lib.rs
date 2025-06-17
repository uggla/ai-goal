// src/lib.rs
use anyhow::{Context, Result, bail};
use futures::future::join_all;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::chat::ChatMessageResponse;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::completion::GenerationResponse;
use ollama_rs::generation::completion::request::GenerationRequest;
use reqwest::Client;
use serde_json::Value;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tiktoken_rs::cl100k_base;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::utils::{build_output, format_duration_hhmmss};

// Define a struct for model information
#[derive(Debug, Clone)]
struct ModelInfo {
    name: &'static str,
    url: &'static str,
}

// Define model URLs and expected filenames using the struct
const MODEL_FILES: [ModelInfo; 3] = [
    ModelInfo {
        name: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
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
    println!("Checking system prerequisites.");

    check_ffmpeg_availability()?;

    let http_client = Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("Could not build HTTP client")?;

    check_and_download_models(&http_client).await?;
    check_ollama_api_and_model(&http_client).await?;

    Ok(())
}

pub fn convert_to_wav_mono_16k<P: AsRef<Path>>(
    input: P,
    root_dir: P,
    force: bool,
) -> Result<PathBuf> {
    let input_path = input.as_ref();

    println!("Convert audio file to meet whisper requirements.");

    if !input_path.exists() {
        bail!("Input file does not exist: {}", input_path.display());
    }

    let input_filename = match input_path.file_stem() {
        Some(filename) => filename.to_string_lossy(),
        None => bail!("Invalid file name"),
    };

    let output = build_output(
        root_dir,
        "audio",
        &format!("{}_mono16k.wav", input_filename),
    )?;

    if output.exists() && !force {
        info!("{} already exists.", output.path.display());
        return Ok(output.path);
    }

    let status = Command::new("ffmpeg")
        .args([
            "-i",
            input_path.to_str().unwrap(),
            "-ac",
            "1",
            "-ar",
            "16000",
            output.path.to_str().unwrap(),
            "-y",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("Failed to execute ffmpeg")?;

    if !status.success() {
        bail!("ffmpeg fails with status: {}", status);
    }

    Ok(output.path)
}

pub fn transcribe_audio<P: AsRef<Path>>(
    audio_path: P,
    output_dir: P,
    model_path: P,
    model_name: &str,
    n_threads: Option<u8>,
    language: Option<String>,
) -> Result<PathBuf> {
    let model_path = match model_path.as_ref().to_str() {
        Some(path) => path,
        None => bail!("Invalid file name"),
    };

    let transcript_dir = output_dir
        .as_ref()
        .join(format!("transcript_{}", model_name));

    let n_threads = n_threads.unwrap_or(num_cpus::get() as u8);

    println!("Transcribe audio file using {n_threads} threads.");

    fs::create_dir_all(&transcript_dir).context(format!(
        "Cannot create output folder {}",
        transcript_dir.display()
    ))?;

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
        "medium" => whisper_rs::DtwMode::ModelPreset {
            model_preset: whisper_rs::DtwModelPreset::Medium,
        },
        _ => bail!("Unknow model"),
    };

    let ctx = WhisperContext::new_with_params(model_path, context_param)
        .context("Failed to load model")?;
    // Create a state
    let mut state = ctx.create_state().context("Failed to create state")?;

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(n_threads.into());
    // Enable translation.
    if language.is_some() {
        params.set_translate(true);
    } else {
        params.set_translate(false);
    }
    params.set_language(language.as_deref());
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Open the audio file.
    let reader = hound::WavReader::open(&audio_path).context(format!(
        "Failed to open file {}",
        audio_path.as_ref().display()
    ))?;

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
        .collect::<Result<_, _>>()
        .context("Failed to read samples from WAV file")?;

    let mut audio = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).context("Conversion error")?;

    if channels != 1 {
        bail!(">1 channels unsupported");
    }

    if sample_rate != 16000 {
        bail!("Sample rate must be 16KHz");
    }

    // Run the model.
    state
        .full(params, &audio[..])
        .context("Failed to run model")?;

    // Create a file to write the transcript to.
    let transcript_path = transcript_dir.join("transcript.txt");
    let mut file = File::create(&transcript_path).with_context(|| {
        format!(
            "Failed to create transcript file: {}",
            transcript_path.display()
        )
    })?;

    // Iterate through the segments of the transcript.
    let num_segments = state
        .full_n_segments()
        .context("failed to get number of segments")?;
    for i in 0..num_segments {
        let segment = match state.full_get_segment_text(i) {
            Ok(s) => s,
            Err(_) => "<unreadable utf-8>".to_string(),
        };

        let start_timestamp = state
            .full_get_segment_t0(i)
            .context("failed to get start timestamp")?;
        let end_timestamp = state
            .full_get_segment_t1(i)
            .context("failed to get end timestamp")?;

        // Format the segment information as a string.
        // let line = format!(
        //     "[{:.3} - {:.3}]: {}\n",
        //     start_timestamp as f32 / 100.0,
        //     end_timestamp as f32 / 100.0,
        //     segment
        // );

        let start_timestamp = Duration::from_secs_f32(start_timestamp as f32 / 100.0);
        let end_timestamp = Duration::from_secs_f32(end_timestamp as f32 / 100.0);
        let line = format!(
            "[{} - {}]: {}\n",
            format_duration_hhmmss(start_timestamp),
            format_duration_hhmmss(end_timestamp),
            segment
        );

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .context("failed to write to file")?;
    }
    Ok(transcript_path)
}

pub async fn summarize_file_with_ollama<P: AsRef<Path>>(
    model: &str,
    transcript_path: P,
    output_dir: P,
) -> Result<PathBuf> {
    let transcript_path = transcript_path.as_ref();
    let output_dir = output_dir.as_ref().join(format!("summary_{}", model));
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Cannot create summary output folder: {}",
            output_dir.display()
        )
    })?;

    const MAX_TOKENS: usize = 1500;
    let content = fs::read_to_string(transcript_path).with_context(|| {
        format!(
            "Failed to read transcript file: {}",
            transcript_path.display()
        )
    })?;

    let tokenizer = cl100k_base()?;

    let tokens = tokenizer.encode_with_special_tokens(&content);
    dbg!(&tokens);
    let mut summaries = Vec::new();
    let mut history = vec![];

    let mut client = Ollama::default();
    for chunk in tokens.chunks(MAX_TOKENS) {
        let chunk_text = tokenizer.decode(chunk.to_vec())?;

        let messages = vec![
            ChatMessage::system(
                "Tu es un assistant qui résume un texte dans sa langue d'origine et de manière consise.".to_string(),
            ),
            ChatMessage::user(format!(
                "Voici un extrait de texte à résumer :\n{chunk_text}"
            )),
        ];

        let res = client
            .send_chat_messages_with_history(
                &mut history,
                ChatMessageRequest::new(model.to_string(), messages),
            )
            .await;

        if let Ok(res) = res {
            summaries.push(res.message.content.trim().to_string());
        }
    }

    let final_summary = if summaries.len() == 1 {
        summaries.remove(0)
    } else {
        let merged = summaries.join("\n");
        let messages = vec![
            ChatMessage::system("Tu es un assistant de résumé.".to_string()),
            ChatMessage::user(format!(
                "Voici plusieurs résumés partiels :\n{merged}\nFais un résumé global."
            )),
        ];
        let res = client
            .send_chat_messages_with_history(
                &mut history,
                ChatMessageRequest::new(model.to_string(), messages),
            )
            .await;

        if let Ok(res) = res {
            res.message.content.trim().to_string()
        } else {
            "".to_string()
        }
    };

    let summary_path = output_dir.join("summary.txt");

    fs::write(&summary_path, final_summary)
        .with_context(|| format!("Failed to write summary to: {}", summary_path.display()))?;
    dbg!(summaries);
    Ok(summary_path)
}

// const MAX_TOKENS: usize = 2048;
//
// pub async fn summarize_text(text: &str, model: &str) -> Result<String> {
//     let client = Ollama::default()?;
//     let tokenizer = cl100k_base()?; // compatible avec GPT/Mistral/LLaMA
//
//     let tokens = tokenizer.encode_with_special_tokens(text);
//     let mut summaries = Vec::new();
//
//     for chunk in tokens.chunks(MAX_TOKENS) {
//         let chunk_text = tokenizer.decode(chunk.to_vec())?;
//
//         let messages = vec![
//             ChatMessage::system("Tu es un assistant qui résume un texte dans sa langue d'origine."),
//             ChatMessage::user(format!(
//                 "Voici un extrait de texte à résumer :\n{chunk_text}"
//             )),
//         ];
//
//         let req = ChatRequest::new(model.to_string(), messages);
//         let res = client.chat(req).await?;
//         summaries.push(res.message.content.trim().to_string());
//     }
//
//     let final_summary = if summaries.len() == 1 {
//         summaries.remove(0)
//     } else {
//         let merged = summaries.join("\n");
//         let messages = vec![
//             ChatMessage::system("Tu es un assistant de résumé."),
//             ChatMessage::user(format!(
//                 "Voici plusieurs résumés partiels :\n{merged}\nFais un résumé global concis."
//             )),
//         ];
//         let req = ChatRequest::new(model.to_string(), messages);
//         let res = client.chat(req).await?;
//         res.message.content.trim().to_string()
//     };
//
//     Ok(final_summary)
// }
