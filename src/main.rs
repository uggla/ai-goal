use std::process::Command;

use anyhow::Context;
use clap::Parser;
use simple_logger::SimpleLogger;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optional flag to enable debug. -ddd = trace, -dd = debug, -d = info, 0 = error.
    #[arg(short = 'd', action = clap::ArgAction::Count)]
    debug: u8,

    /// Input file argument (required)
    input_file: String,
}

fn check_input_file(input_file: &str) -> anyhow::Result<()> {
    Ok(())
}

fn check_external_tools() -> anyhow::Result<()> {
    let output = Command::new("ffmepeg")
        .arg("-version")
        .output()
        .context("ffmepeg not found")?;

    // if !output.status.success() {
    //     error_chain::bail!("Command executed with failing error code");
    // }
    //
    Ok(())
}

fn main() -> anyhow::Result<()> {
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

    check_external_tools()?;
    check_input_file(&cli.input_file)?;
    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_fake() {
        assert_eq!(1, 1);
    }

    #[test]
    fn test_check_external_tools() {
        assert!(check_external_tools().is_ok());
    }
}
