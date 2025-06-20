use ai_goal::{
    check_all_system_requirements, convert_to_wav_mono_16k, summarize_file_with_ollama,
    transcribe_audio,
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
    /// Output language for the transcription
    #[arg(short = 'l', long = "lang")]
    lang: Option<String>,
    /// Number of thread to use for whisper. Default to number of cpu cores.
    #[arg(short = 't', long = "thread")]
    thread: Option<u8>,
    /// Force recreate the output
    #[arg(short = 'f', long = "force")]
    force: bool,
    /// Input file argument (required)
    input_file: String,
    /// Output folder to place output files (required)
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

    let model_path = "models/ggml-base.bin";
    let model_name = "base";

    let transcript_path = match transcribe_audio(
        audio_path,
        PathBuf::from(&cli.output_dir),
        PathBuf::from(model_path),
        model_name,
        cli.thread,
        cli.lang,
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

    let model = "granite3.3:2b"; // ou llama3, gemma, etc.

    let _summary =
        match summarize_file_with_ollama(model, transcript_path, PathBuf::from(&cli.output_dir))
            .await
        {
            Ok(path) => {
                info!("✅ Summary saved to \"{}\".", path.display());
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
