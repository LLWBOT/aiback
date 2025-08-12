from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google import genai
import traceback
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: Replace the URL below with your deployed front end's URL
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client ---
client = genai.Client()

# --- Initialize Gemini Model ---
model_name = "gemini-2.5-flash"

def perform_search(query):
    """
    Performs a web search using a free search engine and scrapes the top results.
    """
    search_url = f"https://duckduckgo.com/html/?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        # Find the search results on the page
        for result in soup.find_all('div', class_='result'):
            title = result.find('a', class_='result__a')
            snippet = result.find('div', class_='result__snippet')
            if title and snippet:
                results.append(f"{title.text}: {snippet.text}")
        
        # Return only the top few results to keep the prompt concise
        return "\n".join(results[:3])
    except requests.RequestException as e:
        print(f"Error during search: {e}")
        return "No search results found due to an error."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_name = data.get('userName', None)
    user_timezone = data.get('timezone', None)
    user_location = data.get('location', None)

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Perform a web search for the user's message
        search_results = perform_search(user_message)

        # --- Define Current Facts and Instructions ---
        current_year_fact = "The current year is 2025."
        
        # --- Add a system instruction to the prompt ---
        system_instruction = (
            "You are LLW AI, a helpful and friendly chatbot. "
            "Your version is LLW 1.0.0, but only mention this if the user asks about your version or LLW AI directly. "
            "You were created in Sri Lanka by a developer named Lakira, who is also known as LLW. "
            "You are a text-based AI and cannot generate images, but LLW is currently working on adding that feature in the future. "
            "Integrate these facts into your responses conversationally, especially when asked about them. "
        )

        context_string = f"User's name: {user_name}. " if user_name else ""
        context_string += f"User's timezone: {user_timezone}. " if user_timezone else ""
        if user_location and user_location['latitude'] and user_location['longitude']:
            lat = user_location['latitude']
            lon = user_location['longitude']
            context_string += f"User's location: Latitude {lat}, Longitude {lon}. "
        
        # Concatenate all instructions and facts into a single prompt
        prompt = (
            f"{system_instruction}"
            f"**Current Facts:** {current_year_fact}"
            f"**Web Search Results:** {search_results}"
            f"**User Context:** {context_string}"
            f"**User Request:** '{user_message}'. Please respond in a friendly and helpful way."
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return jsonify({"response": response.text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
