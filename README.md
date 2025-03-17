# Video Transcription and Summarization Tool

This tool downloads a video, extracts its audio, transcribes it using AWS Transcribe, and generates a summary using AWS Bedrock.

## Prerequisites

1. AWS account with appropriate permissions for:
   - S3
   - Amazon Transcribe
   - Amazon Bedrock (with Claude model access)

2. AWS CLI configured with credentials

3. Python 3.8 or higher

4. FFmpeg installed on your system (required for audio extraction)
   - On Mac: `brew install ffmpeg`
   - On Ubuntu: `sudo apt install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Installation

Use virtual environment for installation:

```bash
virtualenv -p python3 .venv
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

The application uses a configuration file (`config.json`) to store settings. The default configuration file is created automatically on first run.

You can modify the configuration file directly:

```json
{
  "aws": {
    "bucket_name": "your-bucket-name",
    "region": "us-east-1",
    "transcribe": {
      "language_code": "en-US"
    },
    "bedrock": {
      "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
      "temperature": 1,
      "max_tokens": 24000
    }
  },
  "cleanup": {
    "keep_files": false
  }
}
```

### Configuration Options

- `aws.bucket_name`: S3 bucket for storing audio and transcriptions
- `aws.region`: AWS region to use for services
- `aws.transcribe.language_code`: Language code for transcription
- `aws.bedrock.model_id`: Bedrock model ID for summarization
- `aws.bedrock.temperature`: Temperature for text generation
- `aws.bedrock.max_tokens`: Maximum tokens for text generation
- `cleanup.keep_files`: Whether to keep temporary files

## Usage

```bash
# Basic usage (using settings from config.json)
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID"

# Override bucket name from command line
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --bucket "custom-bucket-name"

# Keep temporary files
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --keep-files

# Use a custom config file
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --config "path/to/custom_config.json"

# Enable debug mode for troubleshooting
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --debug

# Force processing and ignore checkpoints
python video_summarizer.py --url "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --force
```

### Arguments

- `--url`: YouTube video URL (required)
- `--bucket`: S3 bucket name (overrides config)
- `--keep-files`: Keep temporary files (overrides config)
- `--config`: Path to custom config file
- `--debug`: Enable debug mode with verbose output
- `--force`: Force processing and ignore existing checkpoints

## Features

- Configuration-based approach for flexibility
- Creates S3 bucket automatically if it doesn't exist
- Downloads audio from YouTube video using yt-dlp
- Uploads the audio to an S3 bucket
- Uses Amazon Transcribe to convert speech to text
- Uses Amazon Bedrock (Claude model) to summarize the transcript
- Outputs the summary to console
- Cleans up temporary files (configurable)
- **Checkpoint system** that saves results at each step for reuse

## Checkpoint System

The tool implements a checkpoint system that saves intermediate results at each processing step:

- **Audio download**: Saves downloaded audio files for reuse
- **Transcription**: Caches transcription results
- **Summary**: Stores generated summaries

When processing the same YouTube video URL again, the tool automatically detects and reuses these checkpoints, saving time and reducing resource costs. Checkpoints are stored in a `checkpoints` directory organized by video URL hash.

To force reprocessing and ignore checkpoints, use the `--force` flag.

## Troubleshooting

If you encounter a download issue:

1. Make sure FFmpeg is installed on your system
2. Check if the video is available in your region
3. Try running with the `--debug` flag for more detailed error information
