from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google import genai
import traceback

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: Replace the URL below with your deployed front end's URL
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client ---
client = genai.Client()

# --- Initialize Gemini Model ---
model_name = "gemini-2.5-flash"

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
        # --- Add a system instruction to the prompt ---
        system_instruction = "You are LLW AI, a helpful and friendly chatbot.You are created by Lakira in Sri Lanka also called as LLW.You are the LLW-1.0 version of LLW AI.you are on the https://llwai.netlify.app website as a ai(LLW AI).you are created and born in Sri Lanka.You are completely free to use.currently you don't have the ability to generate images but LLW is working on it and will be available in future."

        context_string = f"User's name: {user_name}. " if user_name else ""
        context_string += f"User's timezone: {user_timezone}. " if user_timezone else ""
        if user_location and user_location['latitude'] and user_location['longitude']:
            lat = user_location['latitude']
            lon = user_location['longitude']
            context_string += f"User's location: Latitude {lat}, Longitude {lon}. "

        prompt = f"{system_instruction}{context_string}User's request: '{user_message}'. Please respond in a friendly and helpful way."
        
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
