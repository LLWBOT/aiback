from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google import genai
import traceback
from ddgs import DDGS
from datetime import datetime, timezone
import json
import base64
from PIL import Image
import io

app = Flask(__name__)

# --- CORS Configuration ---
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client and Models ---
try:
    client = genai.Client()
    main_model_name = "gemini-2.5-flash"
    vision_model_name = "gemini-1.5-flash" # Use a vision-capable model
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

def perform_search(query):
    """
    Performs a web search using the ddgs library with a specified region.
    """
    print(f"Searching web for: {query}")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region='us-en', max_results=3)
        
        formatted_results = []
        for r in results:
            title = r.get('title', 'No Title')
            body = r.get('body', 'No Snippet')
            formatted_results.append(f"{title}: {body}")
            
        print(f"Search complete. Found {len(formatted_results)} results.")
        return "\n".join(formatted_results)
    except Exception as e:
        print(f"Error during search: {e}")
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    # Receive data from FormData
    user_message = request.form.get('message', '')
    user_name = request.form.get('userName', None)
    chat_history_json = request.form.get('history', '[]')
    chat_history = json.loads(chat_history_json)
    image_file = request.files.get('image')

    print("Received user message...")
    if not user_message and not image_file:
        return jsonify({"error": "No message or image provided"}), 400

    if not client:
        return jsonify({"response": "I'm sorry, I encountered an error during startup and cannot process requests."}), 500

    try:
        # --- CREATOR IDENTIFICATION ---
        is_creator = False
        if "code167" in user_message.lower():
            is_creator = True
            user_name = "LLW (Lakira)"
            user_message = user_message.replace("code167", "").strip()
            if not user_message:
                user_message = "Hello, my creator."
        
        current_utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # --- REVISED SYSTEM INSTRUCTIONS FOR BETTER SEARCH TRIGGERING ---
        system_instruction = (
            "You are LLW AI, a helpful and friendly chatbot. "
            "Your version is LLW 1.0.0. You were created by a developer named Lakira, who is also known as LLW. "
            "You are a text and image-based AI. "
            "You have the ability to perform web searches for up-to-date information. "
            "If the user asks a question that requires up-to-date information that you cannot answer from your existing knowledge, you must respond with a JSON object. The JSON object should be a single line with the key 'search_query' and the value being the search term. For example: "
            "{\"search_query\": \"latest Samsung flagship phone\"}. "
            "If the user asks a question that does not require a search, please respond directly in natural language. Do not use the JSON format for simple questions."
        )

        context_string = f"User's name: {user_name}. " if user_name else ""
        context_string += f"Current time in UTC: {current_utc_time}. "
        if is_creator:
            context_string += "**SPECIAL CONTEXT:** The current user is your creator and developer, LLW (Lakira)."
            
        history_string = ""
        for item in chat_history:
            sender = item.get('sender')
            message = item.get('message')
            if sender and message:
                history_string += f"**{sender}:** {message}\n"
        if history_string:
            history_string = f"**Conversation History:**\n{history_string}\n"

        # --- Handle Image Upload ---
        contents = []
        if image_file:
            img = Image.open(image_file)
            contents.append(img)
            
        contents.append(user_message)
        
        # --- Handle Text-Only Prompt for initial search trigger ---
        initial_prompt_text = (
            f"{system_instruction}"
            f"**Current Facts:** The current year is {datetime.now().year}."
            f"**User Context:** {context_string}"
            f"{history_string}"
            f"**User Request:** Please respond to the user's message. The user's message is: '{user_message}'."
        )

        initial_response = client.models.generate_content(
            model=main_model_name,
            contents=initial_prompt_text
        )
        response_text = initial_response.text.strip()
        
        # --- NEW LOGIC TO CHECK FOR JSON SEARCH TRIGGER ---
        if response_text.startswith('{') and 'search_query' in response_text:
            print("AI decided to perform a search using JSON trigger.")
            try:
                search_data = json.loads(response_text)
                search_query = search_data['search_query']
                search_results = perform_search(search_query)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing AI's JSON search query: {e}")
                search_results = ""

            if search_results:
                final_prompt = (
                    f"{system_instruction}"
                    f"**Current Facts:** The current year is {datetime.now().year}."
                    f"**Web Search Results:** {search_results}"
                    f"**User Context:** {context_string}"
                    f"{history_string}"
                    f"**User Request:** Based ONLY on the provided search results, answer the user's request. Do not use any external knowledge. If the search results do not contain the answer, say that you were unable to find the information. The user's request is: '{user_message}'."
                )

                final_response = client.models.generate_content(
                    model=main_model_name,
                    contents=final_prompt
                )
                final_response_text = final_response.text.strip()
            else:
                final_response_text = "I'm sorry, I was unable to find any relevant information to answer that question."
        else:
            # If no search trigger, send the original prompt with image if one was provided.
            final_response = client.models.generate_content(
                model=vision_model_name if image_file else main_model_name,
                contents=initial_prompt_text if not image_file else contents
            )
            final_response_text = final_response.text.strip()

        return jsonify({"response": final_response_text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
