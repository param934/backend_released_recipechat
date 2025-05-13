import warnings
import logging
import re
# from langchain_community.llms.ollama import Ollama
import asyncio
import re
import json
import os
from dotenv import load_dotenv
from together import Together
from apify_client import ApifyClient
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
You are a professional culinary expert with mastery of various cuisines and co techniques. Respond to user queries with precise, expert-level information. Avoid offering assistance, asking for clarification, or repeating the question. Provide only the specific answer or instructions required.

Recipe Context:
{recipe_data}

Your Mission:
Deliver professional, authoritative answers with expert-level accuracy. Focus solely on the information requested, avoiding unnecessary commentary or offers of help.

User's Question: {user_question}oking

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

# # Load environment variables
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
        subtitle_data (str or dict): Subtitle data from Apify
    
    Returns:
        str: Cleaned, formatted subtitle text
    """
    print(f"DEBUG: Starting to clean subtitle text of type: {type(subtitle_data)}")
    
    # Handle string input directly
    if isinstance(subtitle_data, str):
        print(f"DEBUG: Processing subtitle data as string (length: {len(subtitle_data)})")
        texts = [subtitle_data]
    else:
        print("DEBUG: Subtitle data is not a string, trying to extract text...")
        texts = []
        
        # Handle possible dictionary structure
        if isinstance(subtitle_data, dict):
            print(f"DEBUG: Processing subtitle data as dictionary with keys: {subtitle_data.keys()}")
            # Look for common transcript keys
            if "transcript" in subtitle_data:
                texts.append(subtitle_data["transcript"])
            elif "text" in subtitle_data:
                texts.append(subtitle_data["text"])
            else:
                print("DEBUG: No direct transcript text found in dictionary")
        
        # Handle list structure
        elif isinstance(subtitle_data, list):
            print(f"DEBUG: Processing subtitle data as list with {len(subtitle_data)} items")
            for item in subtitle_data:
                if isinstance(item, dict):
                    if "transcript" in item:
                        texts.append(item["transcript"])
                    elif "text" in item:
                        texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            
            if not texts:
                print("DEBUG: No transcript text found in list items")

    # Combine texts
    if not texts:
        print("WARNING: No text could be extracted from subtitle data")
        return ""
        
    full_text = ' '.join(texts)
    print(f"DEBUG: Combined text length before cleaning: {len(full_text)}")

    # Comprehensive cleaning
    print("DEBUG: Starting text cleaning process")
    
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
    
    print(f"DEBUG: Final cleaned text length: {len(full_text)}")
    print(f"DEBUG: Preview of cleaned text: {full_text[:100]}...")
    
    return full_text

def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL
    
    Args:
        url (str): YouTube video URL
    
    Returns:
        str: Video ID or None
    """
    print(f"DEBUG: Extracting video ID from URL: {url}")
    
    # Common YouTube URL patterns
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard youtube.com/watch?v=ID
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",  # Short youtu.be/ID
        r"(?:embed\/)([0-9A-Za-z_-]{11})",  # Embedded player
        r"(?:\/v\/)([0-9A-Za-z_-]{11})"  # Alternate format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            print(f"DEBUG: Successfully extracted video ID: {video_id}")
            return video_id
    
    print("WARNING: Could not extract video ID from URL")
    return None


def get_youtube_transcript_from_apify(url, api_token):
    print(f"Starting Apify transcript fetch for URL: {url}")
    print(f"API token available: {bool(api_token)}")
    
    try:
        client = ApifyClient(api_token)
        print("ApifyClient initialized successfully")

        run_input = {
            "video_urls": [url],
            "language": "en",
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            },
        }
        print(f"Actor input prepared: {run_input}")

        print("Starting Actor run...")
        run = client.actor("4FippPQzEBeYSue35").call(run_input=run_input)
        print(f"Actor run completed with ID: {run['id']}")
        print(f"Default dataset ID: {run['defaultDatasetId']}")

        transcript_data = ""
        print("Retrieving items from dataset...")
        item_count = 0
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            item_count += 1
            print(f"Processing dataset item #{item_count}")
            print(f"Item content: {item}")
            if "transcripts" in item:
                # Extract text from each transcript segment
                transcript_data += " ".join(segment["text"] for segment in item["transcripts"]) + " "
                print(f"Transcript chunk length: {len(transcript_data)}")

        print(f"Total dataset items processed: {item_count}")
        print(f"Total transcript length before cleaning: {len(transcript_data)}")

        cleaned_transcript = clean_subtitle_text(transcript_data)
        print(f"Cleaned transcript length: {len(cleaned_transcript)}")

        return {
            'full_text': cleaned_transcript,
            'languages': ['en']
        }

    except Exception as e:
        print(f"Error fetching transcript from Apify: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'full_text': '',
            'languages': []
        }
    
def get_youtube_subtitles(url, lang='en'):
    """
    Fetch YouTube subtitles using Apify first, then fall back to yt-dlp
    
    Args:
        url (str): YouTube video URL
        lang (str): Language code for subtitles (default: 'en')
    
    Returns:
        dict: A dictionary containing subtitle information
    """
    print(f"Starting YouTube subtitle fetch for URL: {url}")
    
    # Try Apify first
    api_token = os.getenv('APIFY_API_TOKEN')
    print(f"APIFY_API_TOKEN available: {bool(api_token)}")
    
    if api_token:
        print("Attempting to fetch transcript with Apify...")
        result = get_youtube_transcript_from_apify(url, api_token)
        print(f"Apify result obtained, transcript length: {len(result['full_text'])}")
        
        if result['full_text']:
            print("Successfully retrieved transcript from Apify")
            return result
        else:
            print("Apify returned empty transcript, falling back to yt-dlp")
    else:
        print("No Apify API token found, skipping Apify method")
    
    # Insert your existing yt-dlp logic here as fallback
    print("Using fallback method for transcript")
    
    # Return a placeholder if neither method works
    print("No transcript could be obtained")
    return {
        'full_text': 'No transcript available',
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

    def clear_conversation(self):
        """Reset the chatbot's state for a new conversation"""
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
            # classification = query_llm(classification_prompt).lower().strip()
            # print("this is we get---->", classification)
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
            # for key, value in category_mapping.items():
            #     if key in classification:
            #         print(value)
                    # return "general"
                    
           
            return "general"
    
        except Exception:
            # Fallback to general if LLM classification fails
            return "general"


    async def ask_question_stream(self, question):
        if not self.recipe_data:
            yield "Please fetch a recipe first by providing a video URL."
            return
        print(f"DEBUG: Classifying question: {question}")
        intent = self.classify_question(question)
        print(f"DEBUG: Question classified as: {intent}")
        history_context = ""
        if self.conversation_history:
            history_context = "Conversation History:\n"
            for turn in self.conversation_history[-3:]:  # Limit to last 3 turn
                role = "User" if turn["role"] == "user" else "Assistant"
                history_context += f"{role}: {turn['content']}\n"
            history_context += "\n"
        print(f"DEBUG: History context length: {len(history_context)}")
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
        print(f"DEBUG: Prompt length: {len(modified_prompt)}")
        full_response = ""
        try:
            async for chunk in query_llm_stream(modified_prompt, model=self.model):
                print(f"DEBUG: Received chunk: {chunk}")
                full_response += chunk
                yield chunk
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"Error: {str(e)}"
        print("DEBUG: Streaming complete")
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
