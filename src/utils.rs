use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};

pub struct Output {
    pub path: PathBuf,
}

impl Output {
    pub fn exists(&self) -> bool {
        match fs::exists(&self.path) {
            Ok(true) => true,
            Ok(false) => false,
            // Not sure that's a good approach
            Err(_) => false,
        }
    }
}

pub fn format_duration_hhmmss(d: Duration) -> String {
    let secs = d.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

pub fn build_output<P: AsRef<Path>>(
    root_dir: P,
    output_dir: &str,
    filename: &str,
) -> Result<Output> {
    let output_path = root_dir.as_ref().join(output_dir);
    fs::create_dir_all(&output_path)
        .with_context(|| format!("Cannot create output folder {}", output_path.display()))?;

    let file_path = output_path.join(filename);

    Ok(Output { path: file_path })
}
