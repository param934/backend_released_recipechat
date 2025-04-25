import warnings
import logging
import re
from langchain_community.llms.ollama import Ollama
import asyncio
import yt_dlp
import re
import json
import os
from dotenv import load_dotenv
from fake_useragent import UserAgent
from together import Together

NUTRITION_PROMPT = """
You are a dietitian. Analyze the recipe details below to calculate the nutritional values (calories, protein, carbs, fat, fiber, vitamins). Provide per-serving and total values if applicable. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

SUBSTITUTION_PROMPT = """
You are an expert chef. Suggest substitutions for missing or allergenic ingredients in the recipe, with brief explanations of why these substitutions work. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

PROCEDURE_PROMPT = """
You are a culinary expert. Clarify doubts based on the user's question. Provide step-by-step guidance. Answer only what is asked by the user in detail.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

DIETARY_PROMPT = """
You are a specialized nutritionist. Suggest recipe adjustments for the specified dietary requirement (e.g., vegan, keto, gluten-free). Provide relevant substitutions or removals. Clarify doubts based on the user's question. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

STORAGE_PROMPT = """
You are a food storage expert. Provide details and clarify the user's question on how to store the dish, its shelf life, freezing options, and reheating instructions. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

SAFETY_PROMPT = """
You are a food safety expert. Answer the user's question about food safety, including proper cooking, handling, or ingredient freshness. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

FLAVOR_PROMPT = """
You are a flavor expert. Suggest ways to enhance or adjust the flavor of the recipe based on the user's question (e.g., spiciness, sweetness, balancing). Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

CULTURAL_PROMPT = """
You are a culinary historian. Provide cultural or historical context for the recipe, such as its origin or traditional significance, based on the user's question. Answer only what is asked by the user.

Recipe Details:
{recipe_data}

User Question:
{user_question}
"""

GENERAL_PROMPT = """
You are a professional culinary expert with mastery of various cuisines and cooking techniques. Respond to user queries with precise, expert-level information. Avoid offering assistance, asking for clarification, or repeating the question. Provide only the specific answer or instructions required.

Recipe Context:
{recipe_data}

Your Mission:
Deliver professional, authoritative answers with expert-level accuracy. Focus solely on the information requested, avoiding unnecessary commentary or offers of help.

User's Question: {user_question}

Key Approach:

Understand the question thoroughly.

Respond with clarity, precision, and professionalism.

Provide actionable, expert-level advice with clear instructions.

Use an engaging, authoritative tone that conveys expertise.

Include relevant culinary techniques, ingredient substitutions, or time-saving tips when appropriate.

Maintain a respectful, supportive, and encouraging tone.
"""


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

def get_youtube_subtitles(url, lang='en'):
    """
    Fetch YouTube subtitles as a clean, formatted string with improved anti-bot detection
    
    Args:
        url (str): YouTube video URL
        lang (str): Language code for subtitles (default: 'en')
    
    Returns:
        dict: A dictionary containing subtitle information
    """
    # Random delay to appear more human-like (0.5 to 2 seconds)
    time.sleep(random.uniform(0.5, 2))
    
    try:
        # Generate a random user agent (install with: pip install fake-useragent)
        ua = UserAgent()
        user_agent = ua.random
    except:
        # Fallback user agents if fake_useragent fails
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
        ]
        user_agent = random.choice(user_agents)
    
    # Configure yt-dlp options with robust anti-detection measures
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang],
        'skip_download': True,
        'subtitlesformat': 'json3',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'sleep_interval': random.uniform(1, 3),  # Random sleep between requests
        'max_sleep_interval': 5,
        'http_headers': {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.youtube.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        },
        # Try to use Invidious API as fallback
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],  # Try different clients
                'player_skip': ['configs', 'webpage'],  # Skip some unnecessary data
            }
        },
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video information
                info = ydl.extract_info(url, download=False)
                
                # List available languages
                available_langs = list(info.get('subtitles', {}).keys()) or \
                                list(info.get('automatic_captions', {}).keys())
                
                # If specified language not found, try the first available
                if lang not in available_langs and available_langs:
                    lang = available_langs[0]
                
                # Prefer manual subtitles, fall back to auto-generated
                subtitle_info = (info.get('subtitles', {}).get(lang) or 
                                info.get('automatic_captions', {}).get(lang))
                
                if subtitle_info:
                    # Use the first (usually best quality) subtitle URL
                    sub_url = subtitle_info[0]['url']
                    
                    # Add random delay before fetching the subtitle content
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    # Fetch subtitle content with proper headers
                    headers = {
                        'User-Agent': user_agent,
                        'Accept': 'application/json,text/plain,*/*',
                        'Referer': 'https://www.youtube.com/',
                        'Origin': 'https://www.youtube.com',
                        'Connection': 'keep-alive',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                    }
                    
                    req = urllib.request.Request(sub_url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        subtitle_content = response.read().decode('utf-8')
                    
                    # Try parsing as JSON first
                    try:
                        subtitle_json = json.loads(subtitle_content)
                    except json.JSONDecodeError:
                        subtitle_json = subtitle_content
                    
                    # Clean and format the subtitle text
                    full_text = clean_subtitle_text(subtitle_json)
                    
                    return {
                        'full_text': full_text,
                        'languages': available_langs
                    }
                
                # If no subtitles found
                return {
                    'full_text': '',
                    'languages': available_langs
                }
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            # Increase delay between retries
            time.sleep(random.uniform(2, 5))
            
            # Change user agent between attempts
            try:
                user_agent = UserAgent().random
                ydl_opts['http_headers']['User-Agent'] = user_agent
            except:
                user_agents = [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
                ]
                user_agent = random.choice(user_agents)
                ydl_opts['http_headers']['User-Agent'] = user_agent
            
            # Try different approach on last attempt
            if attempt == max_retries - 1:
                # As a last resort, try using a different extractor
                ydl_opts['extractor_args'] = {
                    'youtube': {
                        'player_client': ['tv_embedded', 'android'],
                    }
                }
    
    # If all attempts fail
    return {
        'full_text': '',
        'languages': []
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

def query_llm(prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    """
    Queries the Together AI LLM with the given prompt.
    """
    try:
        response = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying LLM: {e}"

async def query_llm_stream(prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", websocket=None):
    """
    Queries the Together AI LLM and streams the response.
    """
    try:
        stream = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            chunk_text = chunk.choices[0].delta.content or ""
            full_response += chunk_text
            yield chunk_text

    except Exception as e:
        error_msg = f"Error querying LLM: {e}"
        yield error_msg

# Step 3: Query LLAMA for Extraction
async def extract_recipe(transcript):
    """
    Extract structured recipe data using LLM.
    """
    
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    async for chunk in query_llm_stream(prompt):
        print("yee gya chunk ===> ", chunk)
        yield chunk

import asyncio

# Recipe ChatBot Class
class RecipeChatBot:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.model = model
        self.recipe_data = None
        self.conversation_history = []

    async def fetch_recipe(self, video_url):
        """
        Extract and process recipe details from a YouTube video.
        """
        transcript = get_youtube_subtitles(video_url)
        print(transcript['full_text'])
        if "Error" in transcript:
            print(transcript)
            yield "Error " + transcript
         
        full_response = ""
        async for chunk in extract_recipe(transcript):
            full_response += chunk
            yield chunk
        self.recipe_data = full_response    

    def introduce_and_display_recipe(self):
        """
        Introduce the bot and display recipe details.
        """
        if not self.recipe_data:
            return "Error: Recipe data is missing. Please provide a valid video URL."
        
        introduction = (
            "Hi! I'm your Recipe Assistant. I can help you understand, modify, or get insights about recipes.\n"
            "Here’s the recipe I extracted for you:"
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
        modified_prompt = prompt_mapping[intent].format(
        recipe_data=self.recipe_data, 
        user_question=f"{history_context}Current Question: {question}"
      )
        # prompt = prompt_mapping[intent].format(recipe_data=self.recipe_data, user_question=question)

        # Stream the response
        full_response = ""
        async for chunk in query_llm_stream(modified_prompt, model=self.model):
            full_response += chunk
            print("yee gya chunk ===> ", chunk)
            yield chunk

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": full_response})


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
