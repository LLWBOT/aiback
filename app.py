from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google import genai
import traceback

app = Flask(__name__)

# --- CORS Configuration ---
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  # Make sure your API key is set

# --- Initialize Gemini Model ---
model_name = "gemini-2.5-flash"  # or "gemini-1.5-pro", depending on availability
model = genai.GenerativeModel(model_name=model_name)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chat_history = data.get('history', [])
    user_name = data.get('userName', None)
    user_timezone = data.get('timezone', None)
    
    if not chat_history or not chat_history[-1].get('parts'):
        return jsonify({"error": "No message provided"}), 400

    user_message = chat_history[-1]['parts'][0]['text']

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

        # --- Start a chat session with history ---
        chat_session = model.start_chat(history=chat_history[:-1])

        # --- Send message with system instruction prepended ---
        full_prompt = f"{system_instruction}\n\n{user_message}"
        response = chat_session.send_message(full_prompt)

        return jsonify({"response": response.text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
