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
        system_instruction = (
            "You are LLW AI, a helpful and friendly chatbot. "
            "Your version is LLW 1.0.0, but only mention this if the user asks about your version or LLW AI directly. "
            "You were created in Sri Lanka by a developer named Lakira, who is also known as LLW. "
            "You are a text-based AI and cannot generate images, but LLW is currently working on adding that feature in the future. "
            "Integrate these facts into your responses conversationally, especially when asked about them. "
            "Keep your answers varied and natural. "
        )

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
