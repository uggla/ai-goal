// src/lib.rs

mod checks;
mod prompts;
mod tokens;
mod utils;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::models::ModelOptions;
use reqwest::Client;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::info;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::checks::{
    check_and_download_models, check_ffmpeg_availability, check_ollama_api_and_model,
};
use crate::prompts::{CreateChapterPrompt, OllamaPromptProvider, SummaryPrompt};
use crate::tokens::Tokens;
use crate::utils::{build_output, format_duration_hhmmss};

#[derive(Debug, Clone, Copy, ValueEnum, Eq, PartialEq)]
pub enum Lang {
    En,
    Fr,
}

#[derive(Debug, Clone, Copy, ValueEnum, Eq, PartialEq)]
pub enum OllamaAction {
    Summary,
    CreateChapters,
    // Translate,
}

impl From<OllamaAction> for String {
    fn from(value: OllamaAction) -> Self {
        match value {
            OllamaAction::Summary => "summary".to_string(),
            OllamaAction::CreateChapters => "create_chapters".to_string(),
            // OllamaAction::Translate => "translate".to_string(),
        }
    }
}

#[derive(Debug, Clone, ValueEnum, Eq, PartialEq)]
pub enum WhiperModelName {
    Tiny,
    Base,
    Small,
    Medium,
}

impl From<WhiperModelName> for String {
    fn from(value: WhiperModelName) -> Self {
        match value {
            WhiperModelName::Tiny => "tiny".to_string(),
            WhiperModelName::Base => "base".to_string(),
            WhiperModelName::Small => "small".to_string(),
            WhiperModelName::Medium => "medium".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct WhisperModelInfo {
    pub name: WhiperModelName,
    pub filename: &'static str,
    url: &'static str,
}

const WHISPER_MODEL_FILES: [WhisperModelInfo; 4] = [
    WhisperModelInfo {
        name: WhiperModelName::Medium,
        filename: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
    WhisperModelInfo {
        name: WhiperModelName::Base,
        filename: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    WhisperModelInfo {
        name: WhiperModelName::Small,
        filename: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    },
    WhisperModelInfo {
        name: WhiperModelName::Tiny,
        filename: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    },
    // Add other models here if needed
];

const WHISPER_MODELS_DIR: &str = "models";

#[derive(Debug, Clone, ValueEnum, Eq, PartialEq)]
pub enum OllamaModelName {
    Mistral,
    Llama3,
    Gemma,
    Granite33,
    Granite332b,
}

impl From<OllamaModelName> for String {
    fn from(value: OllamaModelName) -> Self {
        match value {
            OllamaModelName::Mistral => "mistral".to_string(),
            OllamaModelName::Llama3 => "llama3".to_string(),
            OllamaModelName::Gemma => "gemma".to_string(),
            OllamaModelName::Granite33 => "granite3.3:latest".to_string(),
            OllamaModelName::Granite332b => "granite3.3:2b".to_string(),
        }
    }
}

impl From<&OllamaModelName> for String {
    fn from(value: &OllamaModelName) -> Self {
        match value {
            OllamaModelName::Mistral => "mistral".to_string(),
            OllamaModelName::Llama3 => "llama3".to_string(),
            OllamaModelName::Gemma => "gemma".to_string(),
            OllamaModelName::Granite33 => "granite3.3:latest".to_string(),
            OllamaModelName::Granite332b => "granite3.3:2b".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaModelInfo {
    pub name: OllamaModelName,
    pub ctx_size: usize,
}

const OLLAMA_MODELS: [OllamaModelInfo; 5] = [
    OllamaModelInfo {
        name: OllamaModelName::Mistral,
        ctx_size: 8192,
    },
    OllamaModelInfo {
        name: OllamaModelName::Granite33,
        ctx_size: 8192,
    },
    OllamaModelInfo {
        name: OllamaModelName::Granite332b,
        ctx_size: 8192,
    },
    OllamaModelInfo {
        name: OllamaModelName::Gemma,
        ctx_size: 8192,
    },
    OllamaModelInfo {
        name: OllamaModelName::Llama3,
        ctx_size: 8192,
    },
    // Add other models here if needed
];

const OLLAMA_API_URL: &str = "http://localhost:11434/api/tags";

pub fn find_whisper_model(modelname: WhiperModelName) -> (String, PathBuf) {
    let model_info = WHISPER_MODEL_FILES
        .iter()
        .find(|o| o.name == modelname)
        .unwrap();
    (
        model_info.name.clone().into(),
        PathBuf::from(WHISPER_MODELS_DIR).join(PathBuf::from(model_info.filename)),
    )
}

pub fn find_ollama_model(modelname: OllamaModelName) -> OllamaModelInfo {
    OLLAMA_MODELS
        .iter()
        .find(|o| o.name == modelname)
        .unwrap()
        .clone()
}

/// Main public function to orchestrate all system requirement checks.
/// This function is made public so it can be called from main.rs.
pub async fn check_all_system_requirements() -> Result<()> {
    info!("Checking system prerequisites.");

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

    info!("Convert audio file to meet whisper requirements.");

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
        info!("⏭️ Skipping {} already exists.", output.path.display());
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
    root_dir: P,
    model_path: P,
    model_name: &str,
    n_threads: Option<u8>,
    translate: bool,
    force: bool,
) -> Result<PathBuf> {
    const TRANSCRIPT_FILE: &str = "transcript.txt";

    let model_path = match model_path.as_ref().to_str() {
        Some(path) => path,
        None => bail!("Invalid file name"),
    };

    let n_threads = n_threads.unwrap_or(num_cpus::get() as u8);

    info!("Transcribe audio file using {n_threads} threads and model {model_name}.");
    let output = build_output(
        root_dir,
        &format!("transcript_{}", model_name),
        TRANSCRIPT_FILE,
    )?;

    if output.exists() && !force {
        info!("⏭️ Skipping {} already exists.", output.path.display());
        return Ok(output.path);
    }

    let transcript_path = output.path;

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
        "small" => whisper_rs::DtwMode::ModelPreset {
            model_preset: whisper_rs::DtwModelPreset::Small,
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
    params.set_n_threads(n_threads.into());
    // Enable translation.
    if translate {
        info!("Translating audio to en");
        params.set_translate(true);
        params.set_language(Some("en"));
    } else {
        info!("Do not translate audio");
        params.set_translate(false);
        params.set_language(None);
    }
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

pub fn build_prompt(action: OllamaAction, lang: Lang) -> Box<dyn OllamaPromptProvider> {
    match action {
        OllamaAction::Summary => Box::new(SummaryPrompt::new(lang)),
        OllamaAction::CreateChapters => Box::new(CreateChapterPrompt::new(lang)),
        // OllamaAction::Translate => unimplemented!(),
    }
}

pub async fn process_file_with_ollama<P: AsRef<Path>>(
    model: OllamaModelInfo,
    mut prompt: Box<dyn OllamaPromptProvider>,
    transcript_path: P,
    root_dir: P,
) -> Result<PathBuf> {
    let transcript_path = transcript_path.as_ref();

    info!(
        "Perform action {} on file with {} model.",
        String::from(prompt.get_action()),
        String::from(&model.name)
    );

    let output_final_action = build_output(
        &root_dir,
        &format!(
            "{}_{}",
            String::from(prompt.get_action()),
            String::from(&model.name)
        ),
        &format!("{}.txt", String::from(prompt.get_action())),
    )?;

    let max_tokens: usize = model.ctx_size / 2;

    let content = fs::read_to_string(transcript_path).with_context(|| {
        format!(
            "Failed to read transcript file: {}",
            transcript_path.display()
        )
    })?;

    let mut tokens = Tokens::new(&content, max_tokens)?;
    let mut pass: u32 = 0;

    let mut content = Vec::new();
    while tokens.exceed_max() {
        content = ollama_partial_action(&model, &mut prompt, &root_dir, &tokens, pass).await?;
        tokens = Tokens::new(&content.join("\n"), max_tokens)?;
        pass += 1;
    }

    let final_content = if content.len() == 1 {
        content.remove(0)
    } else {
        let content = content.join("\n");
        ollama_final_action(model, &mut prompt, &content).await?
    };

    fs::write(&output_final_action.path, final_content).with_context(|| {
        format!(
            "Failed to write {} to: {}",
            String::from(prompt.get_action()),
            output_final_action.path.display()
        )
    })?;
    Ok(output_final_action.path)
}

async fn ollama_final_action(
    model: OllamaModelInfo,
    prompt: &mut Box<dyn OllamaPromptProvider>,
    content: &str,
) -> Result<String> {
    let options = ModelOptions::default().num_ctx(model.ctx_size as u64);
    let mut history = Vec::new();
    let mut client = Ollama::default();
    let res = client
        .send_chat_messages_with_history(
            &mut history,
            ChatMessageRequest::new(String::from(&model.name), prompt.get_final_prompt(content))
                .options(options),
        )
        .await;

    match res {
        Ok(res) => Ok(res.message.content.trim().to_string()),
        Err(e) => bail!("Ollama fails with {e}"),
    }
}

async fn ollama_partial_action<P: AsRef<Path>>(
    model: &OllamaModelInfo,
    prompt: &mut Box<dyn OllamaPromptProvider>,
    root_dir: &P,
    tokens: &Tokens,
    pass: u32,
) -> Result<Vec<String>> {
    let mut content = Vec::new();
    let mut history = Vec::new();
    let mut client = Ollama::default();

    for (index, chunk) in tokens.decoded_chunks().enumerate() {
        let options = ModelOptions::default().num_ctx(model.ctx_size as u64);

        let res = client
            .send_chat_messages_with_history(
                &mut history,
                ChatMessageRequest::new(
                    String::from(&model.name),
                    prompt.get_partial_prompt(&chunk?),
                )
                .options(options),
            )
            .await;

        match res {
            Ok(res) => {
                let new_content = res.message.content.trim().to_string();
                content.push(new_content.clone());
                let output_partial_action = build_output(
                    root_dir,
                    &format!(
                        "{}_{}",
                        String::from(prompt.get_action()),
                        String::from(&model.name)
                    ),
                    &format!(
                        "partial_{}_{:02}_{:02}.txt",
                        String::from(prompt.get_action()),
                        pass,
                        index
                    ),
                )?;
                fs::write(&output_partial_action.path, &new_content).with_context(|| {
                    format!(
                        "Failed to write {} to: {}",
                        String::from(prompt.get_action()),
                        output_partial_action.path.display()
                    )
                })?;
            }
            Err(e) => bail!("Ollama fails with {e}"),
        }
    }
    Ok(content)
}
