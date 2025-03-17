import os
import time
import json
import boto3
import argparse
import re
import yt_dlp
import hashlib
from botocore.exceptions import NoCredentialsError, ClientError

def get_config_file_path():
    """Get the path to the config file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'config.json')

def load_config():
    """Load configuration from file."""
    config_file = get_config_file_path()
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return create_default_config()
    else:
        # Create default config if it doesn't exist
        config = create_default_config()
        save_config(config)
        return config

def create_default_config():
    """Create a default configuration."""
    return {
        "aws": {
            "bucket_name": "",
            "region": "us-west-2",
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
            "keep_files": False
        }
    }

def save_config(config):
    """Save configuration to file."""
    config_file = get_config_file_path()
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_file}")
    except Exception as e:
        print(f"Error saving config: {e}")

def validate_bucket_name(name):
    """
    Validate and sanitize S3 bucket name according to AWS rules:
    - 3-63 characters long
    - Can contain lowercase letters, numbers, periods, and hyphens
    - Must start and end with letter or number
    - Cannot contain underscores or uppercase letters
    - Cannot be formatted as an IP address
    
    Returns sanitized name or None if can't be fixed.
    """
    if not name:
        return None
        
    # Convert to lowercase
    name = name.lower()
    
    # Replace underscores with hyphens
    name = name.replace('_', '-')
    
    # Replace invalid characters with hyphens
    name = re.sub(r'[^a-z0-9.-]', '-', name)
    
    # Ensure it doesn't start or end with period/hyphen
    name = re.sub(r'^[.-]+', '', name)
    name = re.sub(r'[.-]+$', '', name)
    
    # Check for IP address format (simple check)
    if re.match(r'^[\d.]+$', name):
        name = f"bucket-{name}"
    
    # Ensure length is between 3 and 63
    if len(name) < 3:
        name = f"bkt-{name}"
    if len(name) > 63:
        name = name[:63]
    
    return name

def ensure_bucket_exists(bucket_name, region='us-west-2'):
    """Check if bucket exists, create if it doesn't."""
    original_name = bucket_name
    
    # Validate and sanitize the bucket name
    bucket_name = validate_bucket_name(bucket_name)
    
    if not bucket_name:
        print("Invalid bucket name that cannot be fixed.")
        return False
        
    if bucket_name != original_name:
        print(f"Bucket name '{original_name}' was invalid. Using '{bucket_name}' instead.")
    
    s3_client = boto3.client('s3', region_name=region)
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} already exists")
        return bucket_name  # Return the valid bucket name
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                region = s3_client.meta.region_name
                if region == 'us-west-2':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
                print(f"Bucket {bucket_name} created successfully in {region}")
                return bucket_name  # Return the valid bucket name
            except Exception as create_error:
                print(f"Error creating bucket: {create_error}")
                return False
        else:
            print(f"Error checking bucket: {e}")
            return False

def get_url_hash(url):
    """Generate a unique hash for a YouTube URL."""
    # Extract video ID if it's a YouTube URL to handle different URL formats for same video
    youtube_id_match = re.search(r'(?:v=|youtu\.be\/|embed\/|v\/|watch\?v=)([^&\n?#]+)', url)
    if youtube_id_match:
        video_id = youtube_id_match.group(1)
        return hashlib.md5(video_id.encode()).hexdigest()
    else:
        # Fall back to hashing the entire URL if it's not YouTube or can't extract ID
        return hashlib.md5(url.encode()).hexdigest()

def get_checkpoint_dir():
    """Get or create the checkpoint directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_checkpoint_path(url, step, extension=None):
    """Get path for checkpoint file for specific processing step."""
    url_hash = get_url_hash(url)
    checkpoint_dir = get_checkpoint_dir()
    
    # Create subdirectory for this URL if it doesn't exist
    url_dir = os.path.join(checkpoint_dir, url_hash)
    if not os.path.exists(url_dir):
        os.makedirs(url_dir)
    
    # Return path with appropriate extension
    if extension:
        return os.path.join(url_dir, f"{step}.{extension}")
    return os.path.join(url_dir, step)

def checkpoint_exists(url, step, extension=None):
    """Check if checkpoint exists for the given URL and step."""
    checkpoint_path = get_checkpoint_path(url, step, extension)
    return os.path.exists(checkpoint_path)

def save_checkpoint_data(url, step, data, extension=None):
    """Save checkpoint data to file."""
    checkpoint_path = get_checkpoint_path(url, step, extension)
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2)
            else:
                f.write(str(data))
        print(f"Checkpoint saved: {step}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint {step}: {e}")
        return False

def load_checkpoint_data(url, step, extension=None, is_json=False):
    """Load checkpoint data from file."""
    checkpoint_path = get_checkpoint_path(url, step, extension)
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            if is_json:
                return json.load(f)
            else:
                return f.read()
    except Exception as e:
        print(f"Error loading checkpoint {step}: {e}")
        return None

def save_checkpoint_file(url, step, source_path):
    """Copy a file to the checkpoint directory."""
    if not os.path.exists(source_path):
        return False
        
    checkpoint_path = get_checkpoint_path(url, step, os.path.splitext(source_path)[1][1:])
    try:
        import shutil
        shutil.copy2(source_path, checkpoint_path)
        print(f"File checkpoint saved: {step}")
        return checkpoint_path
    except Exception as e:
        print(f"Error saving file checkpoint {step}: {e}")
        return None

def download_youtube_audio(url, output_path="temp_audio.mp3"):
    """Download audio from a YouTube video using yt-dlp."""
    # Check if we have a checkpoint for this URL
    if checkpoint_exists(url, "audio", "mp3"):
        cached_path = get_checkpoint_path(url, "audio", "mp3")
        cached_title_path = get_checkpoint_path(url, "title", "txt")
        print(f"Using cached audio from previous download: {cached_path}")
        
        if os.path.exists(cached_title_path):
            with open(cached_title_path, 'r', encoding='utf-8') as f:
                video_title = f.read().strip()
        else:
            video_title = "Unknown Title"
            
        # Copy to output path if different
        if cached_path != output_path:
            import shutil
            shutil.copy2(cached_path, output_path)
            
        return output_path, video_title
    
    try:
        # Create a temporary filename without extension for yt-dlp
        temp_filename = output_path.rsplit('.', 1)[0]
        
        # Set up yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False
        }
        
        # Extract video information to get title
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown Title')

        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # The actual output path with extension added by yt-dlp
        actual_output_path = f"{temp_filename}.mp3"
        
        # Rename if the output path differs from what we want
        if actual_output_path != output_path:
            os.rename(actual_output_path, output_path)
        
        if os.path.exists(output_path):
            print(f"Audio downloaded successfully: {output_path}")
            
            # Save as checkpoint
            save_checkpoint_file(url, "audio", output_path)
            save_checkpoint_data(url, "title", video_title, "txt")
            
            return output_path, video_title
        else:
            print("Download completed but file not found")
            return None, None
    except yt_dlp.utils.DownloadError as e:
        print(f"YouTube download error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None, None

def upload_to_s3(file_path, bucket, s3_file_name=None, region='us-west-2'):
    """Upload a file to an S3 bucket."""
    if s3_file_name is None:
        s3_file_name = os.path.basename(file_path)
    
    s3_client = boto3.client('s3', region_name=region)
    try:
        s3_client.upload_file(file_path, bucket, s3_file_name)
        print(f"File uploaded to S3: s3://{bucket}/{s3_file_name}")
        return f"s3://{bucket}/{s3_file_name}"
    except NoCredentialsError:
        print("AWS credentials not available")
        return None
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def transcribe_audio(job_name, s3_uri, bucket, language_code='en-US', region='us-west-2', url=None):
    """Transcribe audio using Amazon Transcribe."""
    # Check if we have a checkpoint for this URL
    if url and checkpoint_exists(url, "transcript", "txt"):
        print("Using cached transcript from previous transcription")
        transcript = load_checkpoint_data(url, "transcript", "txt")
        return {"transcript": transcript}
    
    transcribe_client = boto3.client('transcribe', region_name=region)
    
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp3',
            LanguageCode=language_code,
            OutputBucketName=bucket,
            OutputKey=f"{job_name}.json"
        )
        
        # Wait for the job to complete
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            print("Transcription in progress...")
            time.sleep(10)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            print("Transcription completed!")
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            return transcript_uri
        else:
            print("Transcription failed.")
            return None
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def download_transcript(transcript_uri, region='us-west-2', url=None):
    """Download the transcript from S3."""
    # If this is a dict with transcript key, we're using a checkpoint
    if isinstance(transcript_uri, dict) and "transcript" in transcript_uri:
        return transcript_uri["transcript"]
        
    try:
        # Extract bucket and key from the transcript URI
        if transcript_uri.startswith('https://'):
            # If it's an HTTPS URL from Transcribe
            import boto3
            import urllib.parse
            
            # Parse the S3 URI from the Transcribe output URL
            parsed_url = urllib.parse.urlparse(transcript_uri)
            path_parts = parsed_url.path.strip('/').split('/')
            
            # Get bucket and key from the path
            # Format is typically: /bucket-name/path/to/file.json
            bucket = path_parts[0]
            key = '/'.join(path_parts[1:])
            
            # Use S3 client with credentials
            s3_client = boto3.client('s3', region_name=region)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            transcript_data = json.loads(response['Body'].read().decode('utf-8'))
            
        elif transcript_uri.startswith('s3://'):
            # If it's an S3 URI like s3://bucket/key
            import boto3
            
            # Parse S3 URI
            parts = transcript_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            # Use S3 client with credentials
            s3_client = boto3.client('s3', region_name=region)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            transcript_data = json.loads(response['Body'].read().decode('utf-8'))
            
        else:
            # Try direct request as fallback
            import requests
            response = requests.get(transcript_uri)
            transcript_data = response.json()
        
        transcript_text = transcript_data['results']['transcripts'][0]['transcript']
        print(f"Transcript downloaded: {len(transcript_text)} characters")
        
        # Save as checkpoint if URL is provided
        if url:
            save_checkpoint_data(url, "transcript", transcript_text, "txt")
            
        return transcript_text
        
    except Exception as e:
        print(f"Error downloading transcript: {e}")
        return None

def estimate_token_count(text):
    """
    Estimate the number of tokens in the given text.
    Uses a simple heuristic: ~4 characters per token for English text.
    """
    if not text:
        return 0
    
    # Simple estimation: ~4 characters per token for English
    estimated_tokens = len(text) // 4
    
    return estimated_tokens

def summarize_with_bedrock(text, config, url=None):
    """Summarize text using AWS Bedrock."""
    # Check if we have a checkpoint for this URL
    if url and checkpoint_exists(url, "summary", "txt"):
        print("Using cached summary from previous run")
        summary = load_checkpoint_data(url, "summary", "txt")
        return summary
        
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=config['aws']['region']
        )
        
        # Get Bedrock config
        model_id = config['aws']['bedrock']['model_id']
        max_tokens = config['aws']['bedrock']['max_tokens']
        temperature = config['aws']['bedrock']['temperature']
        
        # Create the prompt
        prompt = f"Please summarize the following transcript into sections and points:\n\n{text}\n\nSummary:"
        
        # Use messaging API format for Claude 3 models
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 16000
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        
        # Extract the summary text from the response content
        summary = ""
        for content_item in response_body.get('content', []):
            if content_item.get('type') == 'text':
                summary = content_item.get('text', '')
                break
        
        # Display thinking output if present and in debug mode
        thinking_content = next((item for item in response_body.get('content', []) 
                               if item.get('type') == 'thinking'), None)
        if thinking_content and 'thinking' in thinking_content:
            print("\nModel's thinking process:")
            print("------------------------")
            print(thinking_content['thinking'][:500] + "..." if len(thinking_content['thinking']) > 500 else thinking_content['thinking'])
            print("------------------------\n")
        
        print("Summary generated successfully")
        
        # Save as checkpoint if URL is provided
        if url and summary:
            save_checkpoint_data(url, "summary", summary, "txt")
            
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def save_results(video_title, transcript, summary, output_dir="results", url=None):
    """Save transcription and summarization results to files."""
    try:
        # Create sanitized filename from video title
        safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
        if not safe_title:
            safe_title = f"video_{int(time.time())}"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Generate file paths
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        transcript_file = os.path.join(output_dir, f"{safe_title}_{timestamp}_transcript.txt")
        summary_file = os.path.join(output_dir, f"{safe_title}_{timestamp}_summary.txt")
        
        # Save transcript
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Save summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Results saved to:")
        print(f"- Transcript: {transcript_file}")
        print(f"- Summary: {summary_file}")
        
        return transcript_file, summary_file
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return None, None
    
def clean_up(file_path, s3_client, bucket, s3_key):
    """Clean up temporary files and S3 objects."""
    try:
        # Remove local file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed local file: {file_path}")
            
        # Delete S3 objects
        s3_client.delete_object(Bucket=bucket, Key=s3_key)
        s3_client.delete_object(Bucket=bucket, Key=f"{s3_key.split('.')[0]}.json")
        print(f"Removed S3 objects")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize YouTube videos")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--bucket", help="S3 bucket name (overrides config)")
    parser.add_argument("--keep-files", action="store_true", help="Keep temporary files (overrides config)")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--force", action="store_true", help="Force processing, ignore checkpoints")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading custom config: {e}")
            config = load_config()
    else:
        config = load_config()
    
    # Override config with command line arguments if provided
    if args.bucket:
        config['aws']['bucket_name'] = args.bucket
    
    if args.keep_files:
        config['cleanup']['keep_files'] = True
    
    # Check if bucket is specified
    bucket_name = config['aws']['bucket_name']
    if not bucket_name:
        print("Error: No bucket specified. Please provide a bucket name using --bucket or in config.json")
        return
    
    # Ensure bucket exists
    valid_bucket_name = ensure_bucket_exists(bucket_name, config['aws']['region'])
    if not valid_bucket_name:
        print("Failed to ensure bucket exists. Exiting.")
        return
    
    # Update config with validated bucket name
    if valid_bucket_name != bucket_name:
        config['aws']['bucket_name'] = valid_bucket_name
        save_config(config)
    
    # Generate a unique job name based on timestamp
    job_name = f"audio-transcription-{int(time.time())}"
    audio_path = f"temp_audio_{int(time.time())}.mp3"
    
    # Process the video
    try:
        url = args.url
        
        # Step 1: Download audio from YouTube (with checkpoint)
        print(f"Downloading audio from: {url}")
        audio_file, video_title = download_youtube_audio(url, audio_path)
        if not audio_file:
            print("Failed to download audio. Please check if the video is available and not restricted.")
            return
            
        # Create a sanitized file name for S3
        s3_key = f"{job_name}.mp3"
        
        # Step 2: Upload to S3
        s3_uri = upload_to_s3(audio_file, valid_bucket_name, s3_key, config['aws']['region'])
        if not s3_uri:
            return
        
        print(f"Audio uploaded to S3: {s3_uri}")
            
        # Step 3: Transcribe the audio using language from config (with checkpoint)
        transcript_uri = transcribe_audio(
            job_name, 
            s3_uri, 
            valid_bucket_name, 
            config['aws']['transcribe']['language_code'],
            config['aws']['region'],
            url=args.force and None or url
        )
        if not transcript_uri:
            return
            
        if not isinstance(transcript_uri, dict):  # If not from checkpoint
            print(f"Transcription results available at: {transcript_uri}")
        
        # Step 4: Download the transcript
        transcript = download_transcript(transcript_uri, config['aws']['region'], 
                                        url=args.force and None or url)
        if not transcript:
            return
        
        # Step 4.5: Calculate estimated token count
        estimated_tokens = estimate_token_count(transcript)
        print(f"Estimated transcript token count: ~{estimated_tokens} tokens")
        print(f"Maximum tokens set to: {config['aws']['bedrock']['max_tokens']} tokens")
                
        # Step 5: Summarize the transcript using config (with checkpoint)
        summary = summarize_with_bedrock(transcript, config, url=args.force and None or url)
        if not summary:
            return
            
        # Print results
        print("\n" + "="*50)
        print(f"Video Title: {video_title}")
        print("="*50)
        print("SUMMARY:")
        print(summary)
        print("="*50)
        
        # Step 6: Save the results
        saved_transcript, saved_summary = save_results(video_title, transcript, summary)
        
        # Clean up if specified in config
        if not config['cleanup']['keep_files']:
            s3_client = boto3.client('s3')
            clean_up(audio_file, s3_client, valid_bucket_name, s3_key)
        
        return saved_transcript, saved_summary
            
    except Exception as e:
        print(f"An error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
