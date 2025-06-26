# ai-goal: Local AI Processing CLI Tool

ai-goal is a command-line interface (CLI) tool written in Rust designed for
local audio and video processing.

It currently provides functionalities for transcription, summarization, and
chapter creation, leveraging open-source AI models that run on your local
machine. **A key advantage of this local execution is enhanced privacy,
as your data is processed entirely on your machine and never leaves it.**

This project is part of a company initiative ("ai goal") aimed at exploring
and becoming familiar with Artificial Intelligence concepts and tooling
like [Ollama](https://ollama.ai/).

While it uses Rust—a language I deeply enjoy and find rewarding to work
with—it's primarily an experimental and learning-oriented project,
not production-ready code. It was an excellent opportunity to dive into
asynchronous Rust and explore various crates such as tracing, ollama-rs,
and whisper-rs. This foundational work also serves as a base for future
explorations into more advanced AI concepts like agentic AI and RAG
(Retrieval-Augmented Generation).

The code acts as a "plumbing" layer, orchestrating the execution of
transcription and language model tasks in the correct sequence. It currently
lacks comprehensive testing and may benefit from further refactoring.

## Features

ai-goal currently supports the following operations:

1. **Transcription**: Transcribes audio or video input.
   - Supports translation of the transcript to English.
2. **Summarization**: Generates a summary from the transcribed content.
3. **Chapter Creation**: Divides the transcript into logical chapters.

All processing is performed locally using models optimized for CPU inference,
though performance can vary significantly with model size and available
resources. [Whisper](https://github.com/ggerganov/whisper.cpp) is utilized
for robust transcription, and [Ollama](https://ollama.ai/) handles the large
language model (LLM) operations.

The modular design allows for easy switching
of models to compare outputs, though larger models will naturally consume
more memory and impact performance.

## How to Build

To build and run ai-goal, you need the Rust toolchain installed, you can
find the installation instructions on the [Rust programming language official
site](https://www.rust-lang.org/).

ffmpeg and ollama are required external dependencies.

### Clone the Repository

First, clone the ai-goal repository to your local machine:

```bash
git clone https://github.com/uggla/ai-goal.git
cd ai-goal
```

### Install dependencies

#### 1- Install build dependencies

**For Fedora (using dnf):**

```bash
sudo dnf install openssl-devel gcc clang cmake
```

**For Ubuntu/Debian (using apt):**

```bash
sudo apt-get update
sudo apt-get install gcc libssl-dev pkg-config clang cmake
```

#### 2- Install ffmpeg

**For Fedora (using dnf):**

```bash
sudo dnf install ffmpeg
```

**For Ubuntu/Debian (using apt):**

```bash
sudo apt-get install ffmpeg
```

#### 3- Install ollama

**For Fedora 42 (using dnf):**

```bash
sudo dnf install ollama
```

**For Fedora < 42 and Ubuntu/Debian:**

No packages seem available for ollama, You can find installation instructions on the official [Ollama website](https://github.com/ollama/ollama).

However, on most Linux systems, you can install it manually using:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Build the ai-goal Project

Navigate to the project root directory and run:

```bash
cargo build --release
```

This will compile the project in release mode, producing an optimized executable.

## Command Line Parameters

The ai-goal CLI tool accepts the following arguments:

| Argument          | Type              | Default         | Description                                                                                    |
| :---------------- | :---------------- | :-------------- | :--------------------------------------------------------------------------------------------- |
| \-d, \--debug     | u8 (0-2)          | 0               | Sets the debug verbosity level. Higher values provide more detailed output.                    |
| \-t, \--translate | bool              | false           | If set, attempts to translate the transcription to English.                                    |
| \--thread         | Option\<u32\>     | None (auto)     | Specifies the number of threads to use for transcription. If None, it's auto-detected.         |
| \-f, \--force     | bool              | false           | Forces re-processing even if output files already exist.                                       |
| \--wm             | WhisperModel enum | Small           | Specifies the Whisper model to use for transcription (e.g., tiny, base, small, medium, large). |
| \--om             | OllamaModel enum  | Granite332b     | Specifies the Ollama model to use for LLM tasks (e.g., llama2, mistral, granite33).            |
| \<lang\>          | Language enum     | None (Required) | Sets the language for the source file (e.g., En for English).                                  |
| \<action\>        | ActionType enum   | None (Required) | The action to perform: summary, or create-chapters.                                            |
| \<input_file\>    | PathBuf           | (Required)      | The path to the input audio or video file.                                                     |
| \<output_dir\>    | PathBuf           | (Required)      | The directory where output files will be saved.                                                |

## **Example Usage and Output**

To run the tool and generate a summary of a video using specific models:

```bash
cargo run -r -- en summary ./video/algo_faster.webm output/algo_faster --om granite33 --wm small
```

This command will:

- Define the source file is in English.
- Perform a summary action.
- Process the input file ./video/algo_faster.webm.
- Save outputs to the output/algo_faster directory.
- Utilize the granite33 Ollama model for summarization.
- Use the small Whisper model for transcription.

Upon successful execution, the output/algo_faster/ directory will be structured similarly to this:

```
output/algo_faster/
├── audio
│ └── algo_faster_mono16k.wav
├── summary_granite3.3:latest
│ ├── partial_summary_00_00.txt
│ ├── partial_summary_00_01.txt
│ └── summary.txt
└── transcript_small
└── transcript.txt
```

- audio/: Contains the extracted and pre-processed audio file.
- summary_granite3.3:latest/: Contains the generated summary and any partial summaries.
- transcript_small/: Contains the full transcription.

Tool console output example:

```bash
 🦉 rribaud   main  ~  workspace  rust  ai-goal  time taskset -c 0-3 cargo run -r --  -t 4 en summary ./video/algo_faster.webm output/algo_faster --wm small
   Compiling ai-goal v0.1.0 (/home/rribaud/workspace/rust/ai-goal)
    Finished `release` profile [optimized] target(s) in 7.71s
     Running `target/release/ai-goal -t 4 en summary ./video/algo_faster.webm output/algo_faster --wm small`
2025-06-25T14:56:05.916677Z  INFO ai_goal: 🚀 Starting ai-goal version: 0.1.0
2025-06-25T14:56:05.916695Z  INFO ai_goal: Checking system prerequisites.
2025-06-25T14:56:05.916697Z  INFO ai_goal::checks: 1. Checking for ffmpeg...
2025-06-25T14:56:06.019656Z  INFO ai_goal::checks:    ffmpeg found.
2025-06-25T14:56:06.032935Z  INFO ai_goal::checks: 2. Checking Whisper models...
2025-06-25T14:56:06.032959Z  INFO ai_goal::checks:    Model 'ggml-medium.bin' found.
2025-06-25T14:56:06.032967Z  INFO ai_goal::checks:    Model 'ggml-base.bin' found.
2025-06-25T14:56:06.032969Z  INFO ai_goal::checks:    Model 'ggml-small.bin' found.
2025-06-25T14:56:06.032973Z  INFO ai_goal::checks:    Model 'ggml-tiny.bin' found.
2025-06-25T14:56:06.032985Z  INFO ai_goal::checks: 3. Checking Ollama API and target models: {"llama3", "granite3.3:2b", "gemma", "granite3.3:latest", "mistral"}...
2025-06-25T14:56:06.034781Z  INFO ai_goal::checks:    Ollama API is responsive.
2025-06-25T14:56:06.034823Z  INFO ai_goal::checks:    Found target Ollama model(s) on system: {"granite3.3:latest", "llama3", "granite3.3:2b", "gemma", "mistral"}.
2025-06-25T14:56:06.035220Z  INFO ai_goal: ✅ All prerequisites are met.
2025-06-25T14:56:06.035225Z  INFO ai_goal: Convert audio file to meet whisper requirements.
2025-06-25T14:56:06.035261Z  INFO ai_goal: ⏭️ Skipping output/algo_faster/audio/algo_faster_mono16k.wav already exists.
2025-06-25T14:56:06.035263Z  INFO ai_goal: ✅ Audio file converted : "output/algo_faster/audio/algo_faster_mono16k.wav".
2025-06-25T14:56:06.035603Z  INFO ai_goal: Transcribe audio file using 4 threads and model small.
2025-06-25T14:56:06.035613Z  INFO ai_goal: ⏭️ Skipping output/algo_faster/transcript_small/transcript.txt already exists.
2025-06-25T14:56:06.035615Z  INFO ai_goal: ✅ Transcript saved to "output/algo_faster/transcript_small/transcript.txt".
2025-06-25T14:56:06.035618Z  INFO ai_goal: Perform action summary on file with granite3.3:2b model.
2025-06-25T17:27:18.018711Z  INFO ai_goal: ✅ Result of action summary saved to "output/algo_faster/summary_granite3.3:2b/summary.txt".
2025-06-25T17:27:18.018729Z  INFO ai_goal: ✅ ai-goal completes successfully
```

## **Known Issues & Future Improvements**

- **Context Model Size Tuning**: There is currently no automatic or
  user-configurable mechanism to tune the context window size of the language
  models based on the available system memory. This can lead to out-of-memory
  errors with larger models or very long inputs.
