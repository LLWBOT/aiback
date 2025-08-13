from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google import genai
import traceback
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

# --- CORS Configuration ---
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client and Models ---
try:
    client = genai.Client()
    main_model_name = "gemini-2.5-flash"
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

def perform_search(query):
    """
    Performs a more reliable web search using a different DuckDuckGo endpoint.
    """
    print(f"Searching web for: {query}")
    search_url = "https://html.duckduckgo.com/html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    params = {'q': query}
    try:
        response = requests.post(search_url, headers=headers, data=params, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result'):
            title_tag = result.find('h2', class_='result__title')
            snippet_tag = result.find('div', class_='result__snippet')

            if title_tag and snippet_tag:
                title = title_tag.text.strip()
                snippet = snippet_tag.text.strip()
                results.append(f"{title}: {snippet}")

        print(f"Search complete. Found {len(results)} results.")
        return "\n".join(results[:3])
    except requests.RequestException as e:
        print(f"Error during search: {e}")
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_name = data.get('userName', None)
    user_timezone = data.get('timezone', None)
    user_location = data.get('location', None)

    print("Received user message...")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if not client:
        return jsonify({"response": "I'm sorry, I encountered an error during startup and cannot process requests."}), 500

    try:
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

        initial_prompt = (
            f"{system_instruction}"
            f"**Current Facts:** The current year is 2025."
            f"**User Context:** {context_string}"
            f"**User Request:** Please respond to the user's message. If you do not have enough information to provide an accurate answer, you must respond with the exact phrase: 'I need to perform a search for this information.' Otherwise, please answer the question directly. The user's message is: '{user_message}'."
        )

        print("-" * 50)
        print("Initial Prompt Sent to Gemini:")
        print(initial_prompt)
        print("-" * 50)

        initial_response = client.models.generate_content(
            model=main_model_name,
            contents=initial_prompt
        )

        response_text = initial_response.text.strip()
        search_trigger = "I need to perform a search for this information."

        if search_trigger in response_text:
            print("AI decided to perform a search.")
            search_results = perform_search(user_message)
            
            if search_results:
                prompt_prefix = "Based on a quick web search, I found..."
                
                final_prompt = (
                    f"{system_instruction}"
                    f"**Current Facts:** The current year is 2025."
                    f"**Web Search Results:** {search_results}"
                    f"**User Context:** {context_string}"
                    f"**User Request:** Based on the search results I provided, please answer the user's request. The user's request is: '{user_message}'."
                )

                print("-" * 50)
                print("Final Prompt with Search Results Sent to Gemini:")
                print(final_prompt)
                print("-" * 50)

                final_response = client.models.generate_content(
                    model=main_model_name,
                    contents=final_prompt
                )
                final_response_text = f"{prompt_prefix} {final_response.text.strip()}"
            else:
                final_response_text = "I'm sorry, I was unable to find any relevant information to answer that question."
        else:
            final_response_text = response_text

        return jsonify({"response": final_response_text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
