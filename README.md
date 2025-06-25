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

Whisper.cpp is a required external dependency.

### Clone the Repository

First, clone the ai-goal repository to your local machine:

```bash
git clone https://github.com/uggla/ai-goal.git
cd ai-goal
```

### Install whisper.cpp

**For Fedora (using dnf):**

```bash
sudo dnf install whisper.cpp
```

**For Ubuntu/Debian (using apt):**

```bash
sudo apt-get update
sudo apt-get install whisper.cpp
```

_(Note: If whisper.cpp is not directly available via your package manager,
you might need to build it from source. Refer to the official whisper.cpp
repository for detailed build instructions.)_

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

## **Known Issues & Future Improvements**

- **Context Model Size Tuning**: There is currently no automatic or
  user-configurable mechanism to tune the context window size of the language
  models based on the available system memory. This can lead to out-of-memory
  errors with larger models or very long inputs.
