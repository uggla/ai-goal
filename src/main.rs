use ai_goal::{check_all_system_requirements, convert_to_wav_mono_16k, transcribe_audio};
use anyhow::Result;
use clap::Parser;
use simple_logger::SimpleLogger;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optional flag to enable debug. -ddd = trace, -dd = debug, -d = info, 0 = error.
    #[arg(short = 'd', action = clap::ArgAction::Count)]
    debug: u8,

    /// Input file argument (required)
    input_file: String,
    /// Input file argument (required)
    output_dir: String,
    // TODO: Add language
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    SimpleLogger::new()
        .with_level(match cli.debug {
            3.. => log::LevelFilter::Trace,
            2 => log::LevelFilter::Debug,
            1 => log::LevelFilter::Info,
            0 => log::LevelFilter::Error,
        })
        .with_utc_timestamps()
        .init()?;

    log::debug!("cli arguments = {:#?}", &cli);
    match check_all_system_requirements().await {
        Ok(_) => println!("✅ System configuration validated successfully."),
        Err(e) => {
            eprintln!("❌ Configuration error: {:?}", e);
            std::process::exit(1);
        }
    }

    let audio_path = match convert_to_wav_mono_16k(&cli.input_file, &cli.output_dir) {
        Ok(path) => {
            println!("✅ Audio file converted : {}", path.display());
            path
        }
        Err(e) => {
            eprintln!("❌ Erreur : {}", e);
            std::process::exit(1);
        }
    };

    let model_path = "models/ggml-base.bin";
    let model_name = "base";
    let threads = 4;

    let transcript = transcribe_audio(
        audio_path,
        PathBuf::from(&cli.output_dir),
        PathBuf::from(model_path),
        model_name,
        threads,
        None, // Some("fr"),
    )?;
    println!("Transcript saved to {}", transcript.display());

    Ok(())
}
