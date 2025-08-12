from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from genai import GenerativeModel, Client
import traceback

app = Flask(__name__)

# --- CORS Configuration ---
SITE_URL = "https://llwai.netlify.app"
CORS(app, origins=SITE_URL)

# --- Initialize Gemini Client and Model for older library ---
# CORRECTED: 'Client' must be capitalized
client = genai.Client()
model = GenerativeModel("gemini-2.5-flash")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chat_history = data.get('history', [])
    
    if not chat_history or not chat_history[-1].get('parts'):
        return jsonify({"error": "No message provided"}), 400

    try:
        # Pass the entire chat history to the model
        response = model.generate_content(chat_history)
        
        return jsonify({"response": response.text})
    except Exception as e:
        print("An error occurred during Gemini API call:")
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, I encountered an error trying to process that request."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
