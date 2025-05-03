import warnings
import logging
import re
from langchain_community.llms.ollama import Ollama
import asyncio
import os
from dotenv import load_dotenv
from together import Together
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import backoff  # Add this library for exponential backoff
import csv

# Keep existing prompts from the original code
NUTRITION_PROMPT = """
You are a dietitian. Analyze the recipe details below to calculate the nutritional values (calories, protein, carbs, fat, fiber, vitamins). Provide per-serving and total values if applicable. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

# [Keep all the other prompts from the original code]

# Suppress warnings and logging for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

# Initialize Together AI client
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

together_client = Together(api_key=api_key)

def clean_subtitle_text(subtitle_data):
    """
    Thoroughly clean and format subtitle text
    
    Args:
        subtitle_data (str or dict): Subtitle data from yt-dlp
    
    Returns:
        str: Cleaned, formatted subtitle text
    """
    def extract_text_from_json(data):
        """Extract text from JSON-like subtitle data"""
        texts = []
        
        # Handle nested dictionary structure
        if isinstance(data, dict):
            # Look for 'events' key which often contains subtitles
            events = data.get('events', [])
            for event in events:
                if 'segs' in event:
                    texts.extend(seg.get('utf8', '') for seg in event['segs'] if 'utf8' in seg)
        
        # Handle list of dictionaries
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'utf8' in item:
                    texts.append(item['utf8'])
        
        # Handle string input
        elif isinstance(data, str):
            texts = [data]
        
        return texts

    # Extract text
    if isinstance(subtitle_data, str):
        # For raw VTT or other text formats
        texts = [subtitle_data]
    else:
        texts = extract_text_from_json(subtitle_data)

    # Combine texts
    full_text = ' '.join(texts)

    # Comprehensive cleaning
    # Remove JSON-like syntax and brackets
    full_text = re.sub(r'[\{\}\[\]\"]', '', full_text)
    
    # Remove timestamps and time-related markers
    full_text = re.sub(r'\d+:\d+:\d+\.\d+ --> \d+:\d+:\d+\.\d+', '', full_text)
    full_text = re.sub(r'"tStartMs":\d+,"dDurationMs":\d+', '', full_text)
    
    # Remove extra whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Remove newline characters
    full_text = full_text.replace('\n', ' ')
    
    # Remove extra spaces and trim
    full_text = ' '.join(full_text.split())

    return full_text

# Add session tracking to avoid rate limits
from requests.adapters import HTTPAdapter, Retry
import requests

# Create a session with retry capabilityy
def create_session_with_retry():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

# Modified YouTube service function with retry
@backoff.on_exception(backoff.expo, 
                     (Exception),
                     max_tries=5,
                     max_time=300)



def load_proxies_from_csv(csv_file):
    proxies = []
    print("Loading proxies from CSV file...")
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if required columns exist
                if 'ip' not in row or 'port' not in row:
                    print(f"Warning: Row missing required columns: {row}")
                    continue
                
                # Get the protocol from the CSV (default to http if not present or empty)
                protocol = row.get('protocol', 'http').strip().lower() or 'http'
                
                # Check if this proxy supports HTTPS
                supports_https = row.get('allowsHttps', '').lower() == 'true'
                
                # Format proxy string
                proxy_str = f"{protocol}://{row['ip']}:{row['port']}"
                proxy_dict = {"http": proxy_str}
                
                # Only add HTTPS if the proxy supports it
                if supports_https:
                    proxy_dict["https"] = proxy_str
                
                proxies.append(proxy_dict)
                
        print(f"Successfully loaded {len(proxies)} proxies from {csv_file}")
        return proxies
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        return []
    except Exception as e:
        print(f"Error loading proxies: {str(e)}")
        return []

def get_youtube_service():
    SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
    SERVICE_ACCOUNT_FILE = '/secrets/client_secret.json'
    
    # Add delay before creating service
    time.sleep(random.uniform(2, 5))
    print("Creating YouTube service...Service account file: ", SERVICE_ACCOUNT_FILE)
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    return build('youtube', 'v3', credentials=credentials)

def get_video_transcript(video_id):
    youtube = get_youtube_service()  # Your OAuth setup function
    
    try:
        # Add delay before API call
        time.sleep(random.uniform(3, 7))
        print("Fetching captions for video ID:", video_id) 
        # List available caption tracks for the video
        captions_response = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()
        
        # Find English captions, or use the first available
        caption_id = None
        for item in captions_response.get('items', []):
            if item['snippet']['language'] == 'en':
                caption_id = item['id']
                break
        
        # If no English caption found, try to use the first available
        if not caption_id and captions_response.get('items'):
            caption_id = captions_response['items'][0]['id']
        
        if not caption_id:
            return "No captions available for this video."
        
        # Add delay before download request
        time.sleep(random.uniform(3, 7))
        
        # Download the caption track
        subtitle = youtube.captions().download(
            id=caption_id,
            tfmt='srt'  # Format options: srt, sub, sbv, or ttml
        ).execute()
        
        # Process the subtitle content (SRT format) into plain text
        return convert_srt_to_text(subtitle)
        
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def convert_srt_to_text(srt_content):
    """Convert SRT formatted subtitle to plain text"""
    import re
    
    # Remove time codes, indices, and extra whitespace
    text_only = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '\n', srt_content)
    text_only = re.sub(r'^\s*\n', '', text_only, flags=re.MULTILINE)
    
    # Join lines into paragraphs
    paragraphs = []
    current_paragraph = []
    
    for line in text_only.split('\n'):
        if line.strip():
            current_paragraph.append(line.strip())
        elif current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs)

# retry and delay strategies implemented
@backoff.on_exception(backoff.expo, 
                     (Exception), 
                     max_tries=5,
                     max_time=300)
def get_youtube_subtitles(url, lang='en', proxies=None):
    """
    Fetch YouTube subtitles as a clean, formatted string using youtube-transcript-api
    with improved retry and delay mechanism
    
    Args:
        url (str): YouTube video URL
        lang (str): Language code for subtitles (default: 'en')
        proxies (dict): Optional proxy dictionary to route the request
    
    Returns:
        dict: A dictionary containing subtitle information
    """
    import time, random, re
    from youtube_transcript_api import YouTubeTranscriptApi
    
    time.sleep(random.uniform(5, 10))  # Initial delay
    
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return {
            'full_text': '',
            'languages': [],
            'error': 'Invalid YouTube URL'
        }
    
    video_id = video_id_match.group(1)
    max_retries = 5

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = min(30, (2 ** attempt) + random.uniform(0, 1))
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language_code for t in transcript_list]

            time.sleep(random.uniform(2, 5))

            try:
                if lang in available_langs:
                    transcript = transcript_list.find_transcript([lang])
                else:
                    transcript = transcript_list[0]
                    lang = transcript.language_code

                time.sleep(random.uniform(2, 5))
                transcript_data = transcript.fetch()

                full_text = " ".join(entry['text'] for entry in transcript_data).strip()

                return {
                    'full_text': full_text,
                    'languages': available_langs,
                    'transcript_data': transcript_data
                }

            except Exception:
                try:
                    time.sleep(random.uniform(3, 7))
                    for transcript in transcript_list:
                        if transcript.is_generated:
                            transcript_data = transcript.fetch()
                            full_text = " ".join(entry['text'] for entry in transcript_data).strip()
                            return {
                                'full_text': full_text,
                                'languages': available_langs,
                                'transcript_data': transcript_data,
                                'note': 'Using auto-generated transcript'
                            }
                except:
                    pass
                raise

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")

    # Final attempt using proxy if provided
    if proxies:
        try:
            print("Trying to fetch transcript using proxy...")
            time.sleep(random.uniform(10, 15))
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=[lang],
                proxies=proxies
            )
            full_text = " ".join(entry['text'] for entry in transcript_data).strip()
            return {
                'full_text': full_text,
                'languages': [lang],
                'transcript_data': transcript_data,
                'note': 'Retrieved using proxy'
            }
        except Exception as e:
            print(f"Proxy attempt failed: {str(e)}")

    try:
        print("Trying alternative transcript retrieval method...")
        time.sleep(random.uniform(5, 10))
        return {
            'full_text': get_video_transcript(video_id),
            'languages': available_langs if 'available_langs' in locals() else [],
            'note': 'Retrieved using alternative method'
        }
    except Exception as e:
        print(f"Alternative method failed: {str(e)}")

    return {
        'full_text': '',
        'languages': available_langs if 'available_langs' in locals() else [],
        'error': 'Failed to retrieve transcript'
    }
    """
    Fetch YouTube subtitles as a clean, formatted string using youtube-transcript-api
    with improved retry and delay mechanism
    
    Args:
        url (str): YouTube video URL
        lang (str): Language code for subtitles (default: 'en')
    
    Returns:
        dict: A dictionary containing subtitle information
    """
    
    # Substantial initial delay (5-10 seconds)
    time.sleep(random.uniform(5, 10))
    
    # Extract video ID from URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return {
            'full_text': '',
            'languages': [],
            'error': 'Invalid YouTube URL'
        }
    
    video_id = video_id_match.group(1)
    max_retries = 5  # Increased from 3
    
    for attempt in range(max_retries):
        try:
            # Progressive delay between attempts (exponential backoff)
            if attempt > 0:
                delay = min(30, (2 ** attempt) + random.uniform(0, 1))
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            
            # Get available transcripts
            
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language_code for t in transcript_list]
            
            # Add delay before fetching actual transcript
            time.sleep(random.uniform(2, 5))
            
            # Try to get the requested language
            try:
                if lang in available_langs:
                    transcript = transcript_list.find_transcript([lang])
                else:
                    # If requested language not available, get the first available
                    transcript = transcript_list[0]
                    lang = transcript.language_code
                
                # Add delay before fetching transcript data
                time.sleep(random.uniform(2, 5))
                
                # Fetch the transcript data
                transcript_data = transcript.fetch()
                
                # Format full text from transcript entries
                full_text = ""
                for entry in transcript_data:
                    full_text += entry['text'] + " "
                
                full_text = full_text.strip()
                
                return {
                    'full_text': full_text,
                    'languages': available_langs,
                    'transcript_data': transcript_data  # Return the original transcript data too
                }
                
            except Exception as e:
                # If manual transcript fails, try auto-generated
                try:
                    # Add delay before trying auto-generated
                    time.sleep(random.uniform(3, 7))
                    
                    # Get auto-generated transcript if available
                    for transcript in transcript_list:
                        if transcript.is_generated:
                            transcript_data = transcript.fetch()
                            
                            # Format full text from transcript entries
                            full_text = ""
                            for entry in transcript_data:
                                full_text += entry['text'] + " "
                            
                            full_text = full_text.strip()
                            
                            return {
                                'full_text': full_text,
                                'languages': available_langs,
                                'transcript_data': transcript_data,
                                'note': 'Using auto-generated transcript'
                            }
                except:
                    pass
                
                raise e
                
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            
            # If it's the last attempt and still failing
            if attempt == max_retries - 1:
                # As a last resort, try with a proxy if provided
                try:
                    # Add substantial delay before final attempt
                    time.sleep(random.uniform(10, 15))
                    
                    # This would require passing a proxy parameter to the function
                    # But we'll just simulate the concept
                    proxies = None  # Replace with actual proxy if needed
                    if proxies:
                        transcript_data = YouTubeTranscriptApi.get_transcript(
                            video_id, 
                            languages=[lang],
                            proxies=proxies
                        )
                        
                        # Format full text from transcript entries
                        full_text = ""
                        for entry in transcript_data:
                            full_text += entry['text'] + " "
                        
                        full_text = full_text.strip()
                        
                        return {
                            'full_text': full_text,
                            'languages': [lang],  # We don't know other languages when using direct method
                            'transcript_data': transcript_data,
                            'note': 'Retrieved using proxy'
                        }
                except:
                    pass
    
    # If all attempts fail, try a different API approach
    try:
        print("Trying alternative transcript retrieval method...")
        time.sleep(random.uniform(5, 10))
        return {
            'full_text': get_video_transcript(video_id),
            'languages': available_langs if 'available_langs' in locals() else [],
            'note': 'Retrieved using alternative method'
        }
    except Exception as e:
        print(f"Alternative method failed: {str(e)}")
    
    # If everything fails
    return {
        'full_text': '',
        'languages': available_langs if 'available_langs' in locals() else [],
        'error': 'Failed to retrieve transcript'
    }


# Step 2: Recipe Extraction Prompt
EXTRACTION_PROMPT = """
You are a professional chef assistant. Extract and format the following details from the provided recipe transcript. Your output must strictly adhere to the specified structure below. Do not include any additional text, headings, or commentary. Begin the output directly with the recipe title:

\\*\\*Title\\*\\*: The concise name of the recipe.  
\\*\\*Ingredients\\*\\*:  
\\- List all ingredients with their quantities, each preceded by a bullet point (e.g., `\\-`).  
\\*\\*Procedure\\*\\*:  
\\- Step-by-step cooking instructions, each preceded by a bullet point (e.g., `\\-`).  

{transcript}
"""

# Add memory management enhancements
import gc

def query_llm(prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    """
    Queries the Together AI LLM with the given prompt.
    """
    try:
        # Add delay before API call
        time.sleep(random.uniform(1, 3))
        
        response = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Force garbage collection to reduce memory usage
        gc.collect()
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying LLM: {e}"

async def query_llm_stream(prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", websocket=None):
    """
    Queries the Together AI LLM and streams the response.
    """
    try:
        # Add delay before API call
        await asyncio.sleep(random.uniform(1, 3))
        
        stream = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            chunk_text = chunk.choices[0].delta.content or ""
            full_response += chunk_text
            
            # Yield chunks with a small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
            yield chunk_text
            
        # Force garbage collection
        gc.collect()

    except Exception as e:
        error_msg = f"Error querying LLM: {e}"
        yield error_msg

# Step 3: Query LLAMA for Extraction
async def extract_recipe(transcript):
    """
    Extract structured recipe data using LLM.
    """
    
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    
    # Add memory optimization - clear transcript from memory if large
    if len(transcript) > 10000:
        transcript_size = len(transcript)
        transcript = None
        gc.collect()
        print(f"Cleared large transcript ({transcript_size} chars) from memory")
    
    async for chunk in query_llm_stream(prompt):
        print("yee gya chunk ===> ", chunk)
        yield chunk

# Recipe ChatBot Class
class RecipeChatBot:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.model = model
        self.recipe_data = None
        self.conversation_history = []
        # Add memory tracking
        self.memory_usage = []

    async def fetch_recipe(self, video_url):
        """
        Extract and process recipe details from a YouTube video.
        """
        try:
            # Log memory usage
            self.memory_usage.append(self._get_memory_usage())
            
            print(f"Fetching transcript for {video_url}...")
            proxy=load_proxies_from_csv('proxy_list.csv')
            transcript = get_youtube_subtitles(video_url,proxies=proxy)
            
            if not transcript['full_text']:
                error_msg = "Error: Could not retrieve transcript. YouTube API rate limit may have been reached."
                print(error_msg)
                yield error_msg
                return
                
            print(f"Transcript retrieved: {len(transcript['full_text'])} characters")
            
            if "error" in transcript and transcript["error"]:
                print(transcript["error"])
                yield "Error: " + transcript["error"]
                return
                
            # Process transcript in chunks if it's too long
            full_text = transcript['full_text']
            
            # Yield to allow processing time
            await asyncio.sleep(1)
            
            print("Extracting recipe...")
            full_response = ""
            async for chunk in extract_recipe(full_text):
                full_response += chunk
                yield chunk
                
            self.recipe_data = full_response
            
            # Clean up memory
            gc.collect()
            self.memory_usage.append(self._get_memory_usage())
            
        except Exception as e:
            error_msg = f"Error processing recipe: {str(e)}"
            print(error_msg)
            yield error_msg

    def _get_memory_usage(self):
        """Helper method to track memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def introduce_and_display_recipe(self):
        """
        Introduce the bot and display recipe details.
        """
        if not self.recipe_data:
            return "Error: Recipe data is missing. Please provide a valid video URL."
        
        introduction = (
            "Hi! I'm your Recipe Assistant. I can help you understand, modify, or get insights about recipes.\n"
            "Here's the recipe I extracted for you:"
        )
        return f"{introduction}\n\n{self.recipe_data}\n\nFeel free to ask me any questions about the recipe!"

    def classify_question(self, question):
        """
        Intelligently classify the user's question using a more nuanced approach.
        
        Args:
            question (str): The user's input question
        
        Returns:
            str: The most appropriate prompt category
        """
        # Add delay to prevent API rate limits
        time.sleep(random.uniform(0.5, 1.5))
        
        # If no specific category is found, use LLM for intelligent classification
        classification_prompt = f"""
        Classify the following user question into the most appropriate category for a recipe assistant just answer one word of matching category nothing else:

        Question: {question}

        Categories:
        1. nutrition - Questions about calories, nutrients, health
        2. substitution - Ingredient replacements or alternatives
        3. procedure - Cooking methods, steps, techniques, summary
        4. dietary - Diet-specific modifications
        5. storage - Storing, preserving, shelf life
        6. flavor - Taste enhancement, seasoning
        7. safety - Cooking safety, handling
        8. cultural - Recipe origin and history
        9. general - Any other type of question

        Choose the most specific category that matches the question's intent:"""
        
        # Use the LLM to make a final determination
        try:
            classification = query_llm(classification_prompt).lower().strip()
            print("this is we get---->", classification)
            # Map variations to standard categories
            category_mapping = {
                "nutrition": "nutrition",
                "substitute": "substitution",
                "ingredient": "substitution",
                "procedure": "procedure",
                "cooking": "procedure",
                "dietary": "dietary",
                "diet": "dietary",
                "storage": "storage",
                "preserve": "storage",
                "flavor": "flavor",
                "taste": "flavor",
                "safety": "safety",
                "cultural": "cultural",
                "origin": "cultural",
                "general": "general"
            }
            
            # Find the best matching category
            for key, value in category_mapping.items():
                if key in classification:
                    print(value)
                    return "general"
                    
           
            return "general"
    
        except Exception:
            # Fallback to general if LLM classification fails
            return "general"

    async def ask_question_stream(self, question):
        """
        Asynchronous method to generate a streaming response to the user's question.
        
        Args:
            question (str): The user's question about the recipe
        
        Yields:
            str: Chunks of the response as they are generated
        """
        if not self.recipe_data:
            yield "Please fetch a recipe first by providing a video URL."
            return
            
        # Add delay to prevent API rate limits
        await asyncio.sleep(random.uniform(1, 2))
        
        history_context = ""
        if self.conversation_history:
            history_context = "Conversation History:\n"
            for turn in self.conversation_history[-3:]:  # Limit to last 3 turns to prevent prompt overflow
                role = "User" if turn["role"] == "user" else "Assistant"
                history_context += f"{role}: {turn['content']}\n"
            history_context += "\n"
            
        # Determine the appropriate prompt
        intent = self.classify_question(question)
        prompt_mapping = {
            "nutrition": NUTRITION_PROMPT,
            "substitution": SUBSTITUTION_PROMPT,
            "procedure": PROCEDURE_PROMPT,
            "dietary": DIETARY_PROMPT,
            "storage": STORAGE_PROMPT,
            "flavor": FLAVOR_PROMPT,
            "cultural": CULTURAL_PROMPT,
            "safety": SAFETY_PROMPT,
            "general": GENERAL_PROMPT,
        }
        
        # Optimize recipe data if it's too long
        recipe_data = self.recipe_data
        if len(recipe_data) > 5000:
            recipe_data = recipe_data[:5000] + "... (truncated for memory efficiency)"
            
        modified_prompt = prompt_mapping[intent].format(
            recipe_data=recipe_data, 
            user_question=f"{history_context}Current Question: {question}"
        )

        # Stream the response
        full_response = ""
        async for chunk in query_llm_stream(modified_prompt, model=self.model):
            full_response += chunk
            print("yee gya chunk ===> ", chunk)
            
            # Add small delay between chunks to prevent flooding
            await asyncio.sleep(0.01)
            yield chunk

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Clean up memory periodically
        if len(self.conversation_history) > 10:
            # Keep only the last 6 turns
            self.conversation_history = self.conversation_history[-6:]
            gc.collect()

    def display_conversation(self):
        """
        Display the conversation history.
        """
        for turn in self.conversation_history:
            role = turn["role"].capitalize()
            print(f"{role}: {turn['content']}")

async def handle_user_question(user_question):
    async for chunk in bot.ask_question_stream(user_question):
        print(chunk, end='', flush=True)

async def handle_recipe_genrate(url):
    async for chunk in bot.fetch_recipe(url):
        print(chunk, end='', flush=True)

# Main Script
if __name__ == "__main__":
    # Set global timeout for all API calls
    import socket
    socket.setdefaulttimeout(30)  # 30 seconds timeout for all socket operations
    
    # Initialize the bot with more memory-efficient parameters
    bot = RecipeChatBot()

    print("Welcome to the Recipe ChatBot!")
    print("Provide a YouTube link to get started.")

    # Step 1: Fetch Recipe
    video_url = input("Enter YouTube video URL: ").strip()
    asyncio.run(handle_recipe_genrate(video_url))
    print(bot.introduce_and_display_recipe())

    # Step 2: Ask Questions in a Loop
    while True:
        user_question = input("\nYour Question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            print("Thank you for using the Recipe ChatBot! Goodbye.")
            break

        asyncio.run(handle_user_question(user_question))