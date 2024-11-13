import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
import requests
import json
import yt_dlp
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import parse_qs, urlparse
import uvicorn
import traceback
from markdown import markdown
import html
import re

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# YouTube API configuration
DEVELOPER_KEY = os.getenv('YOUTUBE_API_KEY')  # Get API key from environment variables
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Enabling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # It is recommended to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    try:
        logger.debug(f"Extracting video ID from URL: {url}")

        if not url:
            raise ValueError("URL is empty")

        # Handling different YouTube URL formats
        if "youtu.be" in url:
            # Format: https://youtu.be/VIDEO_ID
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com/watch" in url:
            # Format: https://www.youtube.com/watch?v=VIDEO_ID
            query = parse_qs(urlparse(url).query)
            video_id = query.get("v", [None])[0]
        elif "youtube.com/embed" in url:
            # Format: https://www.youtube.com/embed/VIDEO_ID
            video_id = url.split("/")[-1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format")

        if not video_id:
            raise ValueError("Failed to extract video ID")

        logger.debug(f"Extracted video ID: {video_id}")
        return video_id

    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        raise ValueError(f"Invalid YouTube URL: {str(e)}")

def format_transcript_text(text):
    """Format transcript text with proper HTML formatting and structure"""
    if not text or text == "Transcription unavailable":
        return text

    formatted_lines = []
    current_paragraph = []

    lines = text.split('\n')

    for i, line in enumerate(lines):
        if line.strip().startswith('[') and line.strip().endswith(']'):
            if current_paragraph:
                formatted_lines.append(' '.join(current_paragraph) + '<br><br>')
                current_paragraph = []

            formatted_lines.append(f"<br>{line.strip()}<br>")
            continue

        words = line.strip().split()
        if not words:
            continue

        if (len(current_paragraph) > 0 and (
            len(words) < 4 or
            any(words[0].lower().startswith(indicator) for indicator in 
                ['i', 'he', 'she', 'they', 'we', 'but', 'and', 'so', 'well', 'now', 'then'])
        )):
            if current_paragraph:
                last_word = current_paragraph[-1]
                if not any(last_word.endswith(p) for p in '.!?,:;'):
                    current_paragraph[-1] = last_word + '.'
            formatted_lines.append(' '.join(current_paragraph) + '<br><br>')
            current_paragraph = []

        sentence = ' '.join(words)
        if not current_paragraph:
            sentence = sentence[0].upper() + sentence[1:] if sentence else sentence

        current_paragraph.extend(sentence.split())

        if len(current_paragraph) >= 15:
            last_word = current_paragraph[-1]
            if not any(last_word.endswith(p) for p in '.!?,:;'):
                current_paragraph[-1] = last_word + '.'

        if len(current_paragraph) > 50:
            if not any(current_paragraph[-1].endswith(p) for p in '.!?'):
                current_paragraph[-1] = current_paragraph[-1] + '.'
            formatted_lines.append(' '.join(current_paragraph) + '<br><br>')
            current_paragraph = []

    if current_paragraph:
        if not any(current_paragraph[-1].endswith(p) for p in '.!?'):
            current_paragraph[-1] = current_paragraph[-1] + '.'
        formatted_lines.append(' '.join(current_paragraph))

    formatted_text = ''.join(formatted_lines)
    formatted_text = ' '.join(formatted_text.split())

    formatted_text = formatted_text.replace(']<br><br>[', ']<br><br>[')
    formatted_text = formatted_text.replace('] ', ']<br>')
    formatted_text = formatted_text.replace(' [', '<br>[')
    formatted_text = formatted_text.replace('. ', '.<br>')
    formatted_text = formatted_text.replace('! ', '!<br>')
    formatted_text = formatted_text.replace('? ', '?<br>')

    return formatted_text

def get_youtube_transcript(video_id):
    """Get and format YouTube transcript with timestamps by topics"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatted_segments = []
        current_segment = []

        def extract_topic(text, max_length=70):
            cleaned_text = ' '.join(text.split())
            topic = ' '.join(cleaned_text.split()[:10])
            if len(topic) > max_length:
                topic = topic[:max_length] + '...'
            return topic.capitalize()

        # Grouping text into meaningful blocks
        for entry in transcript:
            current_segment.append(entry)

            if (len(current_segment) >= 10 or  
                (len(current_segment) > 1 and 
                 current_segment[-1]['start'] - current_segment[0]['start'] > 30)):

                segment_text = ' '.join(s['text'] for s in current_segment)
                segment_start = current_segment[0]['start']

                hours = int(segment_start // 3600)
                minutes = int((segment_start % 3600) // 60)
                seconds = int(segment_start % 60)

                topic = extract_topic(segment_text)

                formatted_segments.append({
                    'timestamp': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
                    'topic': topic,
                    'content': segment_text,
                })

                current_segment = []

        # Handling the last segment
        if current_segment:
            segment_text = ' '.join(s['text'] for s in current_segment)
            segment_start = current_segment[0]['start']
            hours = int(segment_start // 3600)
            minutes = int((segment_start % 3600) // 60)
            seconds = int(segment_start % 60)
            topic = extract_topic(segment_text)

            formatted_segments.append({
                'timestamp': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
                'topic': topic,
                'content': segment_text,
            })

        # Formatting the final text
        formatted_text = '\n\n'.join(
            f'[{segment["timestamp"]}][TOPIC:{segment["topic"]}]\n{segment["content"]}'
            for segment in formatted_segments
        )

        return formatted_text

    except Exception as e:
        logger.error(f"Error getting transcript: {str(e)}")
        return "Transcription unavailable"

def transcribe_audio_with_whisper(audio_path):
    """Transcribe audio using the Whisper model"""
    try:
        if not os.path.exists(audio_path):
            return "Transcription unavailable"

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        # Adding timestamps and topics to the Whisper transcription
        text = result['text']
        words = text.split()
        formatted_segments = []
        current_position = 0

        for i in range(0, len(words), 30):
            chunk = words[i:i+30]
            timestamp = current_position * 10
            hours = timestamp // 3600
            minutes = (timestamp % 3600) // 60
            seconds = timestamp % 60

            context = ' '.join(chunk[:5])
            formatted_segments.append({
                'timestamp': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
                'topic': context.capitalize() + '...',
                'content': ' '.join(chunk)
            })
            current_position += 1

        formatted_text = '\n\n'.join(
            f'[{segment["timestamp"]}][TOPIC:{segment["topic"]}]\n{segment["content"]}'
            for segment in formatted_segments
        )

        try:
            os.remove(audio_path)
        except Exception:
            pass

        return format_transcript_text(formatted_text)
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return "Transcription unavailable"

def combine_segments_by_topic(text):
    """Group transcript segments by topic and combine timestamps."""
    segments = text.split('\n\n')
    combined_segments = []
    current_topic = None
    current_segments = []
    
    for segment in segments:
        if not segment.strip():
            continue
        
        # Use re.match instead of segment.match and correct patterns
        timestamp_match = re.match(r'\[([\d:]+)\]', segment)
        topic_match = re.match(r'\[TOPIC:(.*?)\]', segment)
        content_match = re.match(r'\[[^\]]*\]\s*([\s\S]*)', segment)
        
        if not (timestamp_match and topic_match and content_match):
            continue
        
        timestamp = timestamp_match.group(1)
        topic = topic_match.group(1).strip()
        content = content_match.group(1).strip()
        
        # Determine if this is a new topic based on semantic similarity
        if current_topic is None or not are_topics_similar(current_topic, topic):
            if current_segments:
                # Combine previous segments
                start_time = current_segments[0]['timestamp']
                end_time = current_segments[-1]['timestamp']
                combined_content = '\n'.join(s['content'] for s in current_segments)
                
                combined_segments.append(f"[{start_time}-{end_time}][TOPIC:{current_topic}]\n{combined_content}")
            
            current_topic = topic
            current_segments = []
        
        current_segments.append({
            'timestamp': timestamp,
            'topic': topic,
            'content': content
        })
    
    # Handle the last group
    if current_segments:
        start_time = current_segments[0]['timestamp']
        end_time = current_segments[-1]['timestamp']
        combined_content = '\n'.join(s['content'] for s in current_segments)
        combined_segments.append(f"[{start_time}-{end_time}][TOPIC:{current_topic}]\n{combined_content}")
    
    return '\n\n'.join(combined_segments)

def are_topics_similar(topic1, topic2):
    """Check if two topics are semantically similar."""
    # Simple word overlap - can be improved using more complex NLP methods
    words1 = set(topic1.lower().split())
    words2 = set(topic2.lower().split())
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    return (overlap / total > 0.3) if total > 0 else False

def translate_text_with_llm(text, target_language):
    """Translate the entire text with thematic structuring"""
    language_config = {
        'en': {
            'name': 'English',
            'prompt': 'Translate to English with thematic grouping:'
        },
        'ru': {
            'name': 'Russian',
            'prompt': 'Переведи на русский язык с тематической группировкой:'
        },
        'fr': {
            'name': 'French',
            'prompt': 'Traduisez en français avec groupement thématique:'
        },
        'es': {  # Added Spanish
            'name': 'Spanish',
            'prompt': 'Traduce al español con agrupación temática:'
        },
        'it': {  # Added Italian
            'name': 'Italian',
            'prompt': 'Traduci in italiano con raggruppamento tematico:'
        },
        'de': {  # Added German
            'name': 'German',
            'prompt': 'Übersetze ins Deutsche mit thematischer Gruppierung:'
        }
    }

    if target_language not in language_config:
        logger.error(f"Unsupported target language: {target_language}")
        return "Unsupported language"

    config = language_config[target_language]

    # Base prompt for translation
    prompt = f'''# Translation Task
{config['prompt']}

Please translate the following transcript to {config['name']}.
Organize the content into logical sections with timestamps and topic headers.

Input Format:
- Text contains timestamp markers [HH:MM:SS]
- Some sections have topic markers [TOPIC:...]

Required Output Format:
- Group related content into 3-5 main sections
- Each section should start with: [HH:MM:SS-HH:MM:SS][TOPIC:Description]
- Preserve chronological order
- Maintain natural flow between sections

Source Text:
{text}

# Important Requirements
1. Keep timestamp ranges accurate
2. Create clear topic descriptions
3. Group related content
4. Translate naturally
5. Preserve proper names
'''

    try:
        logger.debug(f"Preparing translation request for {target_language}")
        logger.debug(f"Text length: {len(text)} chars")
        
        # Prepare data for the request
        request_data = {
            'model': 'llama3.2-vision:11b',
            'prompt': prompt,
            'stream': False,
            'system': f"You are a professional translator specializing in {config['name']} translations."
        }
        
        logger.debug(f"Request data prepared: {json.dumps(request_data)[:200]}...")

        # Send request
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=request_data,
            timeout=60
        )

        # Detailed logging of the response
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            logger.debug(f"Response JSON: {json.dumps(response_json)[:200]}...")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response JSON: {str(e)}")
            logger.debug(f"Raw response content: {response.content[:200]}...")
            return "Translation failed: Invalid response format"

        if response.status_code == 200:
            translated_text = response_json.get('response')
            if not translated_text:
                logger.error("Empty translation in response")
                return "Translation failed: Empty response"
                
            # Check translation format
            if not re.search(r'\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\]', translated_text):
                logger.warning("Missing timestamp ranges in translation")
                formatted_text = format_translation_with_timestamps(translated_text, text)
                return formatted_text
                
            return translated_text
        else:
            error_details = response_json.get('error', 'Unknown error')
            logger.error(f"Translation API error: {response.status_code}, Details: {error_details}")
            return f"Translation failed: API error {response.status_code}"

    except requests.exceptions.Timeout:
        logger.error("Translation request timed out")
        return "Translation failed: Request timed out"
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to LLM service")
        return "Translation failed: Connection error"
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return f"Translation failed: {str(e)}"

def format_translation_with_timestamps(translated_text, original_text):
    """Format translation with timestamps from the original"""
    try:
        # Extract timestamps from the original
        original_timestamps = re.findall(r'\[([\d:]+)\]', original_text)
        
        if not original_timestamps:
            logger.warning("No timestamps found in original text")
            return translated_text
            
        # Split translated text into paragraphs
        paragraphs = translated_text.split('\n\n')
        formatted_paragraphs = []
        
        # Distribute timestamps across paragraphs
        timestamp_index = 0
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            if timestamp_index >= len(original_timestamps) - 1:
                start_time = original_timestamps[timestamp_index]
                end_time = original_timestamps[-1]
            else:
                start_time = original_timestamps[timestamp_index]
                end_time = original_timestamps[min(timestamp_index + 1, len(original_timestamps) - 1)]
                timestamp_index += 1
            
            # Determine the topic based on the first sentence of the paragraph
            first_sentence = re.split(r'[.!?]', paragraph)[0][:50]
            topic = first_sentence.strip() + "..."
            
            formatted_paragraph = f"[{start_time}-{end_time}][TOPIC:{topic}]\n{paragraph}"
            formatted_paragraphs.append(formatted_paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
        
    except Exception as e:
        logger.error(f"Error formatting translation: {str(e)}")
        return translated_text

def get_video_details(video_id):
    """Get detailed information about a video using the YouTube API"""
    try:
        logger.debug(f"Fetching video details for ID: {video_id}")

        if not DEVELOPER_KEY or DEVELOPER_KEY == 'YOUR_YOUTUBE_API_KEY':
            logger.error("YouTube API key is not set")
            raise ValueError("YouTube API key is not set")

        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()

        if not video_response['items']:
            logger.error(f"Video with ID {video_id} not found")
            raise ValueError(f"Video not found: {video_id}")

        video_data = video_response['items'][0]

        # Get related videos from the same channel
        channel_videos_response = youtube.search().list(
            part='snippet',
            channelId=video_data['snippet']['channelId'],
            type='video',
            maxResults=5,
            order='date'
        ).execute()

        # Get video tags and category
        video_tags = video_data['snippet'].get('tags', [])
        video_title_keywords = set(video_data['snippet']['title'].lower().split())
        
        # Search for similar videos from other authors
        similar_videos_response = youtube.search().list(
            part='snippet',
            q=' '.join(list(video_title_keywords)[:3]),  # Use the first 3 keywords
            type='video',
            maxResults=5,
            relevanceLanguage='ru',  # Language can be configured
            videoCategoryId=video_data['snippet'].get('categoryId'),
        ).execute()

        # Filter videos from the same channel
        similar_videos = [{
            'title': item['snippet']['title'],
            'url': f"https://youtube.com/watch?v={item['id']['videoId']}",
            'channel': item['snippet']['channelTitle'],
            'publish_date': item['snippet']['publishedAt'],
            'thumbnail': item['snippet']['thumbnails']['medium']['url']
        } for item in similar_videos_response.get('items', []) 
        if item['snippet']['channelId'] != video_data['snippet']['channelId']][:5]

        channel_videos = [{
            'title': item['snippet']['title'],
            'url': f"https://youtube.com/watch?v={item['id']['videoId']}",
            'publish_date': item['snippet']['publishedAt'],
            'thumbnail': item['snippet']['thumbnails']['medium']['url']
        } for item in channel_videos_response.get('items', []) if item['id']['videoId'] != video_id]

        # Get comments
        try:
            comments_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=50
            ).execute()

            comments_text = "\n".join([
                item['snippet']['topLevelComment']['snippet']['textDisplay']
                for item in comments_response.get('items', [])
            ])
        except Exception as e:
            logger.warning(f"Failed to retrieve comments: {str(e)}")
            comments_text = ""

        comment_summary = get_summary_from_ollama(comments_text) if comments_text else "No comments available"

        return {
            'title': video_data['snippet']['title'],
            'description': video_data['snippet']['description'],
            'view_count': video_data['statistics'].get('viewCount', 'N/A'),
            'like_count': video_data['statistics'].get('likeCount', 'N/A'),
            'comment_count': video_data['statistics'].get('commentCount', 'N/A'),
            'publish_date': video_data['snippet']['publishedAt'],
            'channel_title': video_data['snippet']['channelTitle'],
            'channel_videos': channel_videos,
            'similar_videos': similar_videos,
            'comment_summary': comment_summary
        }

    except HttpError as e:
        logger.error(f"YouTube API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YouTube API error: {str(e)}")

def get_summary_from_ollama(text):
    """Generate a summary using the Ollama API"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2-vision:11b',
                'prompt': f"""Create a detailed analysis of the following text with emotionally intelligent insights:

{text}

Analysis structure:
1. Key Themes and Sentiments:
   - Main topics discussed
   - Overall sentiment and tone
   - Notable patterns in feedback

2. User Engagement Analysis:
   - Common reactions and responses
   - Points of agreement/disagreement
   - Questions and concerns raised

3. Notable Insights:
   - Unique perspectives shared
   - Constructive feedback provided
   - Suggestions for improvement

4. Recommendations:
   - Areas for potential focus
   - Suggested responses to feedback
   - Opportunities for engagement""",
                'stream': False
            }
        )

        if response.status_code == 200:
            return response.json().get('response', 'Summary generation failed')
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return "Summary generation failed"
    except Exception as e:
        logger.error(f"Error getting summary from Ollama: {str(e)}")
        return "Summary generation failed"

def download_youtube_audio(video_url, output_path="audio.mp4"):
    """Download audio from a YouTube video"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path + '.mp3'
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        return None

def format_summary_text(text):
    """Format summary text with proper Markdown and convert to HTML"""
    if not text:
        return ""

    # Convert numbered lists (1., 2., etc.) to proper Markdown
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
            # Ensure proper formatting of lists in Markdown
            formatted_lines.append('\n' + line.strip())
        else:
            formatted_lines.append(line)

    text = '\n'.join(formatted_lines)

    # Convert to HTML with proper formatting
    html_content = markdown(text)

    # Ensure proper spacing between paragraphs
    html_content = html_content.replace('<p>', '<p class="mb-4">')

    return html_content

@app.get("/", response_class=HTMLResponse)
async def root():
    """Send the main application page"""
    return HTML_TEMPLATE

@app.post("/translate")
async def translate_transcript(request: Request):
    """Endpoint to translate transcript with support for splitting into parts"""
    try:
        data = await request.json()
        text = data.get('text')
        target_language = data.get('target_language')

        if not text or not target_language:
            raise HTTPException(status_code=400, detail="Text and target language are required")

        # For long texts, split into parts by timestamps
        if len(text) > 5000:
            chunks = text.split('\n\n')
            translated_chunks = []

            for chunk in chunks:
                if not chunk.strip():
                    continue

                translated_chunk = translate_text_with_llm(chunk, target_language)
                if 'Translation failed' in translated_chunk:
                    raise HTTPException(status_code=500, detail=f"Translation failed for a part")

                translated_chunks.append(translated_chunk)

            translated_text = '\n\n'.join(translated_chunks)
        else:
            translated_text = translate_text_with_llm(text, target_language)

            if 'Translation failed' in translated_text:
                raise HTTPException(status_code=500, detail=translated_text)

        return {"translated_text": translated_text}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_video(request: Request):
    """Analyze a YouTube video and return a comprehensive analysis"""
    try:
        data = await request.json()
        video_url = data.get('url')
        if not video_url:
            raise HTTPException(status_code=400, detail="URL is required")

        video_id = get_video_id(video_url)
        video_info = get_video_details(video_id)

        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get transcript with topics
        transcript = get_youtube_transcript(video_id)

        if transcript == "Transcription unavailable":
            audio_path = download_youtube_audio(video_url)
            if audio_path:
                transcript = transcribe_audio_with_whisper(audio_path)
            else:
                transcript = "Transcription unavailable"

        # Generate summaries and format responses
        if transcript and transcript != "Transcription unavailable":
            content_summary = get_summary_from_ollama(transcript)
            content_summary = format_summary_text(content_summary)
        else:
            content_summary = "Content summary unavailable"

        if 'comment_summary' in video_info:
            video_info['comment_summary'] = format_summary_text(video_info['comment_summary'])

        return {
            'video_info': video_info,
            'transcript': transcript,
            'content_summary': content_summary,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

# HTML template with updated React components
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Video Analyzer Pro</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.5/babel.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: radial-gradient(circle at top right, #1e293b, #0f172a);
            color: #e2e8f0;
            min-height: 100vh;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .gradient-text {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 3rem;
        }

        .input-group {
            display: flex;
            gap: 1rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }

        .input {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.5rem;
            padding: 1rem;
            color: white;
            font-size: 1rem;
            transition: all 0.2s;
        }

        .input:focus {
            outline: none;
            border-color: #60a5fa;
            box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2);
        }

        .button {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .progress-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            height: 4px;
            border-radius: 2px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            transition: width 0.3s ease;
            border-radius: 2px;
            position: relative;
        }

        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 1.5s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 0.75rem;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: transform 0.2s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
        }

        .stat-title {
            color: #94a3b8;
            font-size: 0.875rem;
        }

        .stat-value {
            color: #e2e8f0;
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tabs {
            display: flex;
            gap: 0.5rem;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .tab {
            flex: 1;
            padding: 0.75rem 1.5rem;
            background: transparent;
            color: #94a3b8;
            cursor: pointer;
            border: none;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }

        .tab::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, #60a5fa, #a78bfa);
            opacity: 0;
            transition: opacity 0.2s ease;
            z-index: 0;
        }

        .tab:hover:not(.active) {
            background: rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
        }

        .tab.active {
            color: white;
        }

        .tab.active::before {
            opacity: 1;
        }

        .tab span {
            position: relative;
            z-index: 1;
        }

        .content-area {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0.75rem;
            padding: 1.5rem;
            max-height: 500px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #4b5563 transparent;
        }

        .content-area::-webkit-scrollbar {
            width: 6px;
        }

        .content-area::-webkit-scrollbar-track {
            background: transparent;
        }

        .content-area::-webkit-scrollbar-thumb {
            background-color: #4b5563;
            border-radius: 3px;
        }

        .youtube-player {
            aspect-ratio: 16/9;
            width: 100%;
            border-radius: 0.75rem;
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                        0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .related-video {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: all 0.2s;
        }

        .related-video:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateX(5px);
        }

        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .success-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            color: #86efac;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .spinner {
            animation: spin 1s linear infinite;
            width: 1.25rem;
            height: 1.25rem;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease-out forwards;
        }

        .video-info {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .video-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: white;
        }

        .video-meta {
            display: flex;
            gap: 1rem;
            color: #94a3b8;
            font-size: 0.875rem;
        }

        .transcript-navigation {
            position: sticky;
            top: -30px;
            background: rgba(30, 41, 59, 0.95);
            padding: 1rem;
            margin: -1.5rem -1.5rem 1rem -1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            z-index: 10;
        }

        .timestamp-block {
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 0.75rem;
            border-left: 4px solid #60a5fa;
        }

        .timestamp-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }

        .timestamp {
            color: #60a5fa;
            font-weight: 600;
            font-size: 1.1em;
        }

        .timestamp-topic {
            color: #94a3b8;
            font-weight: 500;
            font-size: 1.1em;
        }

        .timestamp-content {
            margin-top: 0.5rem;
            color: #e2e8f0;
            line-height: 1.6;
        }

        .language-buttons {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .language-button {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .language-button:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .language-button.active {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            border-color: transparent;
        }

        .translation-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            display: inline-block;
            margin-left: 0.5rem;
        }

        .transcript-content {
            line-height: 1.8;
            color: #e2e8f0;
        }

        .transcript-paragraph {
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect, useCallback } = React;

        const StatCard = ({ title, value }) => (
            <div className="stat-card">
                <div className="stat-title">{title}</div>
                <div className="stat-value">{value}</div>
            </div>
        );

        const YouTubePlayer = ({ videoId }) => (
            <div className="youtube-player">
                <iframe
                    width="100%"
                    height="100%"
                    src={`https://youtube.com/embed/${videoId}`}
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                />
            </div>
        );

        // Updated TranscriptTab component with support for additional languages
        const TranscriptTab = ({
            transcript,
            translations,
            setTranslations,
            activeLanguage,
            setActiveLanguage,
            isTranslating,
            setIsTranslating
        }) => {
            const [error, setError] = useState(null);
            
            const languages = [
                { code: 'original', name: 'Original' },
                { code: 'en', name: 'English' },
                { code: 'ru', name: 'Русский' },
                { code: 'es', name: 'Español' },
                { code: 'it', name: 'Italiano' },
                { code: 'de', name: 'Deutsch' },
                { code: 'fr', name: 'Français' }
            ];

            // Split transcript into segments by timestamps
            const splitTranscriptIntoSegments = useCallback((text) => {
                const segments = text.split('\n\n').filter(Boolean);
                return segments.map(segment => {
                    const timestampMatch = segment.match(/\[([\d:]+)\]/);
                    const topicMatch = segment.match(/\[TOPIC:(.*?)\]/);
                    const contentMatch = segment.match(/\][^\]]*\]\s*([\s\S]*)/);
                    
                    return {
                        timestamp: timestampMatch ? timestampMatch[1] : '',
                        topic: topicMatch ? topicMatch[1].trim() : '',
                        content: contentMatch ? contentMatch[1].trim() : segment
                    };
                });
            }, []);

            const translateTranscript = async (targetLang) => {
                // If translation already exists, simply switch to it
                if (translations[targetLang]) {
                    setActiveLanguage(targetLang);
                    return;
                }

                setIsTranslating(true);
                setError(null);
                setActiveLanguage(targetLang);

                try {
                    const response = await fetch('/translate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: transcript,
                            target_language: targetLang
                        }),
                    });

                    if (!response.ok) {
                        const data = await response.json();
                        throw new Error(data.detail || 'Translation failed');
                    }

                    const data = await response.json();
                    // Save translation in the global state
                    setTranslations(prev => ({
                        ...prev,
                        [targetLang]: data.translated_text
                    }));
                    setActiveLanguage(targetLang);
                } catch (error) {
                    setError(`Translation error: ${error.message}`);
                    console.error('Translation error:', error);
                } finally {
                    setIsTranslating(false);
                }
            };

            const formatDisplayText = useCallback((text) => {
                // Check text format
                const isTranslatedFormat = text.includes('-') && text.match(/\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\]/);
                
                if (isTranslatedFormat) {
                    // Translated text format
                    const segments = text.split(/(?=\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\])/);
                    
                    return segments.map((segment, index) => {
                        const timeRangeMatch = segment.match(/\[(\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\]/);
                        const topicMatch = segment.match(/\[TOPIC:(.*?)\]/);
                        const contentMatch = segment.match(/\][^\]]*\]\s*([\s\S]*)/);
                        
                        if (!timeRangeMatch || !topicMatch || !contentMatch) {
                            return null;
                        }

                        return `
                            <div class="timestamp-block">
                                <div class="timestamp-header">
                                    <span class="timestamp">[${timeRangeMatch[1]}]</span>
                                    <span class="timestamp-topic">${topicMatch[1]}</span>
                                </div>
                                <div class="timestamp-content">${contentMatch[1]}</div>
                            </div>
                        `;
                    }).filter(Boolean).join('');
                } else {
                    // Original transcript format
                    const segments = text.split(/(?=\[\d{2}:\d{2}:\d{2}\])/);
                    
                    return segments.map((segment, index) => {
                        const timestampMatch = segment.match(/\[(\d{2}:\d{2}:\d{2})\]/);
                        const topicMatch = segment.match(/\[TOPIC:(.*?)\]/);
                        let content = segment;
                        
                        if (timestampMatch) {
                            content = segment.replace(/\[\d{2}:\d{2}:\d{2}\]/, '');
                        }
                        if (topicMatch) {
                            content = content.replace(/\[TOPIC:.*?\]/, '');
                        }
                        
                        if (!timestampMatch) return null;

                        const topic = topicMatch ? topicMatch[1] : 'Segment ' + (index + 1);

                        return `
                            <div class="timestamp-block">
                                <div class="timestamp-header">
                                    <span class="timestamp">[${timestampMatch[1]}]</span>
                                    <span class="timestamp-topic">${topic}</span>
                                </div>
                                <div class="timestamp-content">${content.trim()}</div>
                            </div>
                        `;
                    }).filter(Boolean).join('');
                }
            }, []);

            const displayText = activeLanguage === 'original' ? 
                transcript : 
                (translations[activeLanguage] || transcript);

            return (
                <div>
                    <div className="transcript-navigation">
                        <div className="language-buttons">
                            {languages.map(lang => (
                                <button
                                    key={lang.code}
                                    onClick={() => lang.code === 'original' ? 
                                        setActiveLanguage('original') : 
                                        translateTranscript(lang.code)
                                    }
                                    className={`language-button ${activeLanguage === lang.code ? 'active' : ''}`}
                                    disabled={isTranslating}
                                >
                                    {lang.name}
                                    {translations[lang.code] && lang.code !== 'original' && (
                                        <span className="translation-status" title="Translation completed" />
                                    )}
                                </button>
                            ))}
                        </div>
                        
                        {isTranslating && (
                            <div className="flex justify-center my-4">
                                <div className="spinner" />
                            </div>
                        )}
                        
                        {error && (
                            <div className="error-message mt-4">
                                {error}
                            </div>
                        )}
                    </div>
                    
                    <div 
                        className="transcript-content"
                        dangerouslySetInnerHTML={{ __html: formatDisplayText(displayText) }}
                    />
                </div>
            );
        };

        // Updated YoutubeAnalyzer component with translation state management
        function YoutubeAnalyzer() {
            const [url, setUrl] = useState('');
            const [loading, setLoading] = useState(false);
            const [analysis, setAnalysis] = useState(null);
            const [error, setError] = useState('');
            const [activeTab, setActiveTab] = useState('summary');
            const [videoId, setVideoId] = useState(null);
            const [progress, setProgress] = useState(0);
            // Add state to store translations
            const [translations, setTranslations] = useState({});
            const [activeLanguage, setActiveLanguage] = useState('original');
            const [isTranslating, setIsTranslating] = useState(false);

            const getVideoIdFromUrl = (url) => {
                try {
                    const urlObj = new URL(url);
                    const searchParams = new URLSearchParams(urlObj.search);
                    return searchParams.get('v');
                } catch (e) {
                    return null;
                }
            };

            const analyzeVideo = async () => {
                if (!url) {
                    setError('Please enter a YouTube URL');
                    return;
                }

                const newVideoId = getVideoIdFromUrl(url);
                if (!newVideoId) {
                    setError('Invalid YouTube URL');
                    return;
                }

                setVideoId(newVideoId);
                setLoading(true);
                setError('');
                setProgress(0);
                setAnalysis(null);
                setTranslations({});
                setActiveLanguage('original');

                const progressInterval = setInterval(() => {
                    setProgress(prev => {
                        if (prev >= 90) {
                            clearInterval(progressInterval);
                            return 90;
                        }
                        return prev + Math.random() * 30;
                    });
                }, 500);

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url }),
                    });

                    if (!response.ok) {
                        const data = await response.json();
                        throw new Error(data.detail || 'Analysis failed');
                    }

                    const data = await response.json();
                    setAnalysis(data);
                    setProgress(100);
                } catch (err) {
                    setError(`Error: ${err.message}`);
                    setProgress(0);
                } finally {
                    clearInterval(progressInterval);
                    setLoading(false);
                }
            };

            const formatNumber = (num) => {
                if (!num || num === 'N/A') return 'N/A';
                return new Intl.NumberFormat().format(num);
            };

            return (
                <div className="container">
                    <h1 className="gradient-text">YouTube Video Analyzer Pro</h1>
                    <p className="subtitle">Deep insights for any YouTube video</p>

                    <div className="input-group">
                        <input
                            type="text"
                            className="input"
                            placeholder="Paste YouTube URL here..."
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && analyzeVideo()}
                        />
                        <button
                            className="button"
                            onClick={analyzeVideo}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <div className="spinner" />
                                    Analyzing...
                                </>
                            ) : 'Analyze Video'}
                        </button>
                    </div>

                    {loading && (
                        <div className="progress-container fade-in">
                            <div 
                                className="progress-bar" 
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    )}

                    {error && (
                        <div className="error-message fade-in">
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9 9v5M9 5h.01" 
                                    stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                            </svg>
                            {error}
                        </div>
                    )}

                    {videoId && analysis && (
                        <div className="fade-in">
                            <div className="card">
                                <YouTubePlayer videoId={videoId} />
                                <div className="video-info">
                                    <h2 className="video-title">{analysis.video_info.title}</h2>
                                    <div className="video-meta">
                                        <span>{analysis.video_info.channel_title}</span>
                                        <span>Published: {new Date(analysis.video_info.publish_date).toLocaleDateString()}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="stats-grid">
                                <StatCard
                                    title="Views"
                                    value={formatNumber(analysis.video_info.view_count)}
                                />
                                <StatCard
                                    title="Likes"
                                    value={formatNumber(analysis.video_info.like_count)}
                                />
                                <StatCard
                                    title="Comments"
                                    value={formatNumber(analysis.video_info.comment_count)}
                                />
                            </div>

                            <div className="card">
                                <div className="tabs">
                                    {['summary', 'transcript', 'comments', 'related'].map((tab) => (
                                        <button
                                            key={tab}
                                            onClick={() => setActiveTab(tab)}
                                            className={`tab ${activeTab === tab ? 'active' : ''}`}
                                        >
                                            <span>
                                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                            </span>
                                        </button>
                                    ))}
                                </div>

                                <div className="content-area">
                                    {activeTab === 'summary' && (
                                        <div>
                                            <h3 className="text-xl font-semibold mb-4">Content Summary</h3>
                                            <div className="success-badge mb-4">AI-generated summary</div>
                                            <div 
                                                className="text-gray-300"
                                                dangerouslySetInnerHTML={{ __html: analysis.content_summary }}
                                            />
                                        </div>
                                    )}

                                    {activeTab === 'transcript' && (
                                        <TranscriptTab
                                            transcript={analysis.transcript}
                                            translations={translations}
                                            setTranslations={setTranslations}
                                            activeLanguage={activeLanguage}
                                            setActiveLanguage={setActiveLanguage}
                                            isTranslating={isTranslating}
                                            setIsTranslating={setIsTranslating}
                                        />
                                    )}

                                    {activeTab === 'comments' && (
                                        <div>
                                            <h3 className="text-xl font-semibold mb-4">Comments Analysis</h3>
                                            <div className="success-badge mb-4">AI-generated analysis</div>
                                            <div 
                                                className="text-gray-300"
                                                dangerouslySetInnerHTML={{ __html: analysis.video_info.comment_summary }}
                                            />
                                        </div>
                                    )}

                                    {activeTab === 'related' && (
                                        <div>
                                            <h3 className="text-xl font-semibold mb-4">Videos from this Channel</h3>
                                            {analysis.video_info.channel_videos.map((video, index) => (
                                                <div key={index} className="related-video">
                                                    <h4 className="font-semibold mb-2">{video.title}</h4>
                                                    <p className="text-gray-400 text-sm mb-2">
                                                        Published: {new Date(video.publish_date).toLocaleDateString()}
                                                    </p>
                                                    <a
                                                        href={video.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-blue-400 hover:text-blue-300 transition-colors inline-flex items-center gap-2"
                                                    >
                                                        Watch Video
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                                            <polyline points="15 3 21 3 21 9" />
                                                            <line x1="10" y1="14" x2="21" y2="3" />
                                                        </svg>
                                                    </a>
                                                </div>
                                            ))}
                                            
                                            <h3 className="text-xl font-semibold mb-4 mt-8">Similar Videos from Other Creators</h3>
                                            {analysis.video_info.similar_videos.map((video, index) => (
                                                <div key={index} className="related-video">
                                                    <h4 className="font-semibold mb-2">{video.title}</h4>
                                                    <p className="text-gray-400 text-sm mb-2">
                                                        Channel: {video.channel}
                                                    </p>
                                                    <p className="text-gray-400 text-sm mb-2">
                                                        Published: {new Date(video.publish_date).toLocaleDateString()}
                                                    </p>
                                                    <a
                                                        href={video.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-blue-400 hover:text-blue-300 transition-colors inline-flex items-center gap-2"
                                                    >
                                                        Watch Video
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                                            <polyline points="15 3 21 3 21 9" />
                                                            <line x1="10" y1="14" x2="21" y2="3" />
                                                        </svg>
                                                    </a>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        ReactDOM.render(<YoutubeAnalyzer />, document.getElementById('root'));
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    