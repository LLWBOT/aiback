from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from datetime import datetime
from google import genai

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: Replace the URL below with your deployed front end's URL
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client ---
# The Client will automatically get the API key from the environment variable
# named GEMINI_API_KEY. You don't need to configure it manually anymore.
client = genai.Client()

# --- Initialize Gemini Model ---
# The model is now accessed through the client.
model_name = "gemini-2.5-flash"
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


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
        context_string = f"User's name: {user_name}. " if user_name else ""
        context_string += f"User's timezone: {user_timezone}. " if user_timezone else ""
        if user_location and user_location['latitude'] and user_location['longitude']:
            lat = user_location['latitude']
            lon = user_location['longitude']
            context_string += f"User's location: Latitude {lat}, Longitude {lon}. "

        prompt = f"User's request: '{user_message}'. {context_string}Please respond in a friendly and helpful way."
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            generation_config=generation_config
        )
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Error generating content from Gemini: {e}")
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
