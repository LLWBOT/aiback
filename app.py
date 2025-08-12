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
    # NEW: Get the full chat history from the request
    chat_history = data.get('history', [])
    user_name = data.get('userName', None)
    user_timezone = data.get('timezone', None)
    
    # NEW: The last message in the history is the user's new message
    user_message = chat_history[-1]['parts'][0]['text'] if chat_history else ""

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
        
        # NEW: The model's conversation now includes the system instruction
        # and the full chat history. The prompt is handled differently.
        chat_session = client.models.start_chat(
            model=model_name,
            history=chat_history[:-1] # Exclude the last message (the new user message)
        )
        
        # The new message is sent to the chat session
        response = chat_session.send_message(user_message)
        
        return jsonify({"response": response.text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
