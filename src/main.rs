use ai_goal::{
    Lang, OllamaAction, OllamaModelName, WhiperModelName, build_prompt,
    check_all_system_requirements, convert_to_wav_mono_16k, find_ollama_model, find_whisper_model,
    process_file_with_ollama, transcribe_audio,
};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::{Level, debug, error, info};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optional flag to enable debug. -ddd = trace, -dd = debug, -d = info, 0 = error.
    #[arg(short = 'd', action = clap::ArgAction::Count)]
    debug: u8,
    /// Translate audio to English.
    #[arg(long = "translate")]
    translate: bool,
    /// Number of thread to use for whisper. Default to number of cpu cores.
    #[arg(short = 't', long = "thread")]
    thread: Option<u8>,
    /// Force recreate the output.
    #[arg(short = 'f', long = "force")]
    force: bool,
    /// Whisper model to use.
    #[arg(long = "wm",value_enum, default_value_t = WhiperModelName::Base)]
    whisper_model: WhiperModelName,
    /// Ollama model to use.
    #[arg(long = "om",value_enum, default_value_t = OllamaModelName::Granite332b)]
    ollama_model: OllamaModelName,

    /// Language of the source audio file.
    lang: Lang,
    /// Action to do
    action: OllamaAction,
    /// Input file argument (required).
    input_file: String,
    /// Output folder to place output files (required).
    output_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");

    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(match cli.debug {
            2.. => Level::TRACE,
            1 => Level::DEBUG,
            0 => Level::INFO,
        })
        .init();

    debug!("cli arguments = {:#?}", &cli);

    let (model_name, model_path) = find_whisper_model(cli.whisper_model);

    info!("🚀 Starting {} version: {}", name, version);
    match check_all_system_requirements().await {
        Ok(_) => info!("✅ All prerequisites are met."),
        Err(e) => {
            error!("❌ Configuration error: {:?}.", e);
            std::process::exit(1);
        }
    }

    let audio_path = match convert_to_wav_mono_16k(&cli.input_file, &cli.output_dir, cli.force) {
        Ok(path) => {
            info!("✅ Audio file converted : \"{}\".", path.display());
            path
        }
        Err(e) => {
            error!("❌ Erreur : {}.", e);
            std::process::exit(1);
        }
    };

    let transcript_path = match transcribe_audio(
        audio_path,
        PathBuf::from(&cli.output_dir),
        model_path,
        &model_name,
        cli.thread,
        cli.translate,
        cli.force,
    ) {
        Ok(path) => {
            info!("✅ Transcript saved to \"{}\".", path.display());
            path
        }
        Err(e) => {
            error!("❌ Erreur : {}.", e);
            std::process::exit(1);
        }
    };

    let prompt = build_prompt(cli.action, cli.lang);

    let _summary = match process_file_with_ollama(
        find_ollama_model(cli.ollama_model),
        prompt,
        transcript_path,
        PathBuf::from(&cli.output_dir),
    )
    .await
    {
        Ok(path) => {
            info!(
                "✅ Result of action {} saved to \"{}\".",
                String::from(cli.action),
                path.display()
            );
            path
        }
        Err(e) => {
            error!("❌ Erreur : {}.", e);
            std::process::exit(1);
        }
    };

    info!("✅ {} completes successfully", name);
    Ok(())
}
