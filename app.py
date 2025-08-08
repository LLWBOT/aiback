from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import random
from datetime import datetime
from spellchecker import SpellChecker
import re
from sympy import sympify, SympifyError

app = Flask(__name__)

# --- New CORS configuration ---
# The provided frontend URL is now added to the allowed origins.
CORS(app, origins=["https://llwai.netlify.app"])

# Initialize SpellChecker
spell = SpellChecker()

# --- Chatbot Training Data ---
training_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("what's up", "greeting"),
    ("hey", "greeting"),
    ("sup", "greeting"),
    ("how are you", "how_are_you"),
    ("how's it going", "how_are_you"),
    ("how's your day", "how_are_you"),
    ("i'm doing great", "how_are_you_response"),
    ("i'm fine", "how_are_you_response"),
    ("all good", "how_are_you_response"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("what is my name", "ask_name"),
    ("who am I", "ask_name"),
    ("tell me my name", "ask_name"),
    ("do you know my name", "ask_name"),
    ("my name is", "provide_name"),
    ("i'm called", "provide_name"),
    ("what is your name", "about_ai"),
    ("who are you", "about_ai"),
    ("what is LLW ai", "about_ai"),
    ("what is the time", "ask_time"),
    ("what time is it", "ask_time"),
    ("do you know the time", "ask_time"),
    ("tell me the time", "ask_time"),
    ("who made you", "about_creator"),
    ("who is your creator", "about_creator"),
    ("who built you", "about_creator"),
    ("i'm happy", "user_feeling_happy"),
    ("i'm feeling great", "user_feeling_happy"),
    ("i'm sad", "user_feeling_sad"),
    ("i'm feeling down", "user_feeling_sad"),
    ("i'm angry", "user_feeling_angry"),
    ("i'm mad", "user_feeling_angry"),
    ("i'm tired", "user_feeling_tired"),
    ("i'm exhausted", "user_feeling_tired"),
    ("i'm frustrated", "user_feeling_frustrated"),
    ("i'm feeling frustrated", "user_feeling_frustrated"),
    ("i'm worried", "user_feeling_worried"),
    ("i'm feeling anxious", "user_feeling_worried"),
    ("i'm confused", "user_feeling_confused"),
    ("i'm lost", "user_feeling_confused"),
    ("i'm bored", "user_feeling_bored"),
    ("i have nothing to do", "user_feeling_bored"),
    ("what is 5 plus 3", "math_solve"),
    ("solve 2 times 7", "math_solve"),
    ("what is 10 divided by 2", "math_solve"),
    ("calculate 15-8", "math_solve"),
    ("can you do math", "math_solve"),
    ("can you calculate", "math_solve"),
    ("what's 4*9", "math_solve"),
    ("2+2", "math_solve"),
]

# --- Preprocessing and Training ---
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def correct_text_spelling(text):
    words = text.lower().split()
    corrected_words = []
    uncorrected_words = []
    
    for word in words:
        correction = spell.correction(word)
        if correction is not None:
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
            if word not in spell.known(words):
                uncorrected_words.append(word)
    
    return " ".join(corrected_words), uncorrected_words

def preprocess_text(text):
    corrected_text, _ = correct_text_spelling(text)
    words = corrected_text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

X_train = [preprocess_text(text) for text, intent in training_data]
y_train = [intent for text, intent in training_data]

model = make_pipeline(TfidfVectorizer(), SVC(probability=True))
model.fit(X_train, y_train)

# --- Dynamic Response Lists ---
about_ai_starters = [
    "I am LLW AI, a friendly language model.",
    "My name is LLW AI.",
    "Think of me as a digital helper built to assist you.",
]

about_ai_purposes = [
    "My purpose is to understand your questions and provide helpful responses.",
    "I was designed to be a conversational assistant.",
    "I'm here to provide information and keep you company.",
]

about_ai_status = [
    "I'm still in development, so I'm always learning new things!",
    "I am a language model still developing, so there can be mistakes and errors.",
    "My understanding is constantly evolving."
]

intent_responses = {
    "greeting": [
        "Hey, {userName}! LLW AI is here to help.",
        "Hi, {userName}! How can I help you today?",
    ],
    "how_are_you": [
        "I'm fine, thanks for asking!",
        "I'm doing well, thank you!",
    ],
    "how_are_you_response": [
        "That's great to hear!",
        "I'm glad you're doing well!",
    ],
    "farewell": [
        "Goodbye! Have a great day.",
        "See you later!",
        "Bye! Take care.",
    ],
    "ask_name": [
        "Your name is {userName}.",
        "You told me your name is {userName}.",
        "I know you as {userName}.",
    ],
    "provide_name": [
        "Thanks, {userName}! I'll remember that.",
        "Hello {userName}, what can I do for you?",
        "Got it, {userName}! Nice to meet you.",
    ],
    "about_creator": [
        "I'm a language model created by the developer known as LLW. I am his passion project!",
        "I was built by LLW as an advanced project to explore the capabilities of artificial intelligence.",
        "My creator is LLW, a developer who enjoys building conversational AIs like me.",
    ],
    "user_feeling_happy": [
        "That's wonderful to hear! What's making you feel so great?",
        "I'm so glad you're happy! Keep that positive energy going.",
        "Awesome! A good mood can make all the difference.",
    ],
    "user_feeling_sad": [
        "I'm sorry to hear that. I'm here if you need to talk.",
        "That sounds tough. It's okay to feel sad sometimes. What's on your mind?",
        "I'm sending you a virtual hug. Hope you feel better soon!",
    ],
    "user_feeling_angry": [
        "That sounds frustrating. What happened to make you feel that way?",
        "It's completely normal to feel angry. Do you want to talk about it?",
        "Take a deep breath. I'm here to listen.",
    ],
    "user_feeling_tired": [
        "Sounds like you could use some rest. Make sure you take a break!",
        "You've been working hard. Maybe a little rest will help you feel refreshed.",
        "Hope you get some rest soon!",
    ],
    "user_feeling_frustrated": [
        "I'm sorry you're feeling frustrated. What can I do to help?",
        "That sounds really annoying. Take a moment to breathe.",
        "It's normal to feel that way. Let's try to figure it out together.",
    ],
    "user_feeling_worried": [
        "I'm sorry to hear you're worried. Can you tell me what's on your mind?",
        "I hope everything works out for you. I'm here to listen if you need it.",
        "It's okay to feel anxious. Just know that I'm here for you.",
    ],
    "user_feeling_confused": [
        "I can understand your confusion. What do you need help with?",
        "Sometimes things can be unclear. Let's try to clear things up.",
    ],
    "user_feeling_bored": [
        "I'm sorry you're bored. Maybe we can find something interesting to talk about?",
        "Boredom can be a tough feeling. What do you enjoy doing?",
        "Let's play a game! Or maybe you have a question I can answer?",
    ],
    "math_solve": [
        "The answer is {result}.",
        "I calculated the answer to be {result}.",
        "The result of your calculation is {result}.",
    ],
}

unknown_responses = [
    "I'm sorry, I can't understand what you said. LLW ai is a language model still developing, so there can be mistakes and errors. Please try again.",
    "My apologies, I didn't quite get that. I'm still learning, and sometimes I make mistakes. Could you please rephrase your message?",
    "It seems I'm having trouble understanding. As a developing AI, my understanding isn't perfect yet. Please try to say that again in a different way.",
]

def generate_ai_description():
    starter = random.choice(about_ai_starters)
    purpose = random.choice(about_ai_purposes)
    status = random.choice(about_ai_status)
    return f"{starter} {purpose} {status}"

def solve_math_problem(text):
    text = text.lower().replace('plus', '+').replace('minus', '-').replace('times', '*').replace('multiplied by', '*').replace('divided by', '/').replace('power of', '**')
    match = re.search(r'[\d\s\(\)\+\-\*/\.]+', text)
    if match:
        expression = match.group(0).strip()
        try:
            result = str(sympify(expression))
            return result
        except SympifyError:
            return "I couldn't solve that math problem. Please check the expression."
    return "I couldn't find a valid math expression in your message."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_name = data.get('userName', None)

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    corrected_text, uncorrected_words = correct_text_spelling(user_message)
    processed_message = preprocess_text(user_message)
    
    predicted_intent = model.predict([processed_message])[0]
    confidence = model.predict_proba([processed_message]).max()

    if predicted_intent == "math_solve":
        math_result = solve_math_problem(user_message)
        if "couldn't solve" in math_result:
            return jsonify({"response": math_result})
        
        response_list = intent_responses.get("math_solve", [])
        if response_list:
            ai_response_template = random.choice(response_list)
            ai_response = ai_response_template.format(result=math_result)
        else:
            ai_response = random.choice(unknown_responses)
        return jsonify({"response": ai_response})

    if predicted_intent == "ask_name" and user_name is None:
        ai_response = "I can't find your name. Please say your name so I can remember it."
        return jsonify({"response": ai_response, "action": "requestName"})

    if predicted_intent == "provide_name" and user_name is None:
        try:
            name_start_index = user_message.lower().find("my name is") + len("my name is")
            extracted_name = user_message[name_start_index:].strip().title()
            if extracted_name:
                ai_response = random.choice(intent_responses["provide_name"]).format(userName=extracted_name)
                return jsonify({"response": ai_response, "foundName": extracted_name})
        except:
            pass

    if predicted_intent == "about_ai":
        ai_response = generate_ai_description()
        return jsonify({"response": ai_response})
    
    if predicted_intent == "ask_time":
        current_time = datetime.now().strftime("%I:%M %p")
        ai_response = f"The current time is {current_time}."
        return jsonify({"response": ai_response})

    if confidence < 0.6:
        if uncorrected_words:
            misspelled_word = uncorrected_words[0]
            ai_response = f"I'm not sure what you mean by \"{misspelled_word}\". Could you please clarify?"
        else:
            ai_response = random.choice(unknown_responses)
    else:
        response_list = intent_responses.get(predicted_intent, [])
        if response_list:
            ai_response_template = random.choice(response_list)
            ai_response = ai_response_template.format(userName=user_name if user_name else 'Friend')
        else:
            ai_response = random.choice(unknown_responses)
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
    ("tell me my name", "ask_name"),
    ("do you know my name", "ask_name"),
    ("my name is", "provide_name"),
    ("i'm called", "provide_name"),
    ("what is your name", "about_ai"),
    ("who are you", "about_ai"),
    ("what is LLW ai", "about_ai"),
    ("what is the time", "ask_time"),
    ("what time is it", "ask_time"),
    ("do you know the time", "ask_time"),
    ("tell me the time", "ask_time"),
    ("who made you", "about_creator"),
    ("who is your creator", "about_creator"),
    ("who built you", "about_creator"),
    ("i'm happy", "user_feeling_happy"),
    ("i'm feeling great", "user_feeling_happy"),
    ("i'm sad", "user_feeling_sad"),
    ("i'm feeling down", "user_feeling_sad"),
    ("i'm angry", "user_feeling_angry"),
    ("i'm mad", "user_feeling_angry"),
    ("i'm tired", "user_feeling_tired"),
    ("i'm exhausted", "user_feeling_tired"),
    ("i'm frustrated", "user_feeling_frustrated"),
    ("i'm feeling frustrated", "user_feeling_frustrated"),
    ("i'm worried", "user_feeling_worried"),
    ("i'm feeling anxious", "user_feeling_worried"),
    ("i'm confused", "user_feeling_confused"),
    ("i'm lost", "user_feeling_confused"),
    ("i'm bored", "user_feeling_bored"),
    ("i have nothing to do", "user_feeling_bored"),
    ("what is 5 plus 3", "math_solve"),
    ("solve 2 times 7", "math_solve"),
    ("what is 10 divided by 2", "math_solve"),
    ("calculate 15-8", "math_solve"),
    ("can you do math", "math_solve"),
    ("can you calculate", "math_solve"),
    ("what's 4*9", "math_solve"),
    ("2+2", "math_solve"),
]

# --- Preprocessing and Training ---
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# This new function now returns the corrected text AND the list of unknown words
def correct_text_spelling(text):
    words = text.lower().split()
    corrected_words = []
    uncorrected_words = []
    
    for word in words:
        correction = spell.correction(word)
        if correction is not None:
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
            # Add the original word to the uncorrected list
            if word not in spell.known(words):
                uncorrected_words.append(word)
    
    return " ".join(corrected_words), uncorrected_words

def preprocess_text(text):
    corrected_text, _ = correct_text_spelling(text)
    words = corrected_text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

X_train = [preprocess_text(text) for text, intent in training_data]
y_train = [intent for text, intent in training_data]

model = make_pipeline(TfidfVectorizer(), SVC(probability=True))
model.fit(X_train, y_train)

# --- Dynamic Response Lists ---
about_ai_starters = [
    "I am LLW AI, a friendly language model.",
    "My name is LLW AI.",
    "Think of me as a digital helper built to assist you.",
]

about_ai_purposes = [
    "My purpose is to understand your questions and provide helpful responses.",
    "I was designed to be a conversational assistant.",
    "I'm here to provide information and keep you company.",
]

about_ai_status = [
    "I'm still in development, so I'm always learning new things!",
    "I am a language model still developing, so there can be mistakes and errors.",
    "My understanding is constantly evolving."
]

intent_responses = {
    "greeting": [
        "Hey, {userName}! LLW AI is here to help.",
        "Hi, {userName}! How can I help you today?",
    ],
    "how_are_you": [
        "I'm fine, thanks for asking!",
        "I'm doing well, thank you!",
    ],
    "how_are_you_response": [
        "That's great to hear!",
        "I'm glad you're doing well!",
    ],
    "farewell": [
        "Goodbye! Have a great day.",
        "See you later!",
        "Bye! Take care.",
    ],
    "ask_name": [
        "Your name is {userName}.",
        "You told me your name is {userName}.",
        "I know you as {userName}.",
    ],
    "provide_name": [
        "Thanks, {userName}! I'll remember that.",
        "Hello {userName}, what can I do for you?",
        "Got it, {userName}! Nice to meet you.",
    ],
    "about_creator": [
        "I'm a language model created by the developer known as LLW. I am his passion project!",
        "I was built by LLW as an advanced project to explore the capabilities of artificial intelligence.",
        "My creator is LLW, a developer who enjoys building conversational AIs like me.",
    ],
    "user_feeling_happy": [
        "That's wonderful to hear! What's making you feel so great?",
        "I'm so glad you're happy! Keep that positive energy going.",
        "Awesome! A good mood can make all the difference.",
    ],
    "user_feeling_sad": [
        "I'm sorry to hear that. I'm here if you need to talk.",
        "That sounds tough. It's okay to feel sad sometimes. What's on your mind?",
        "I'm sending you a virtual hug. Hope you feel better soon!",
    ],
    "user_feeling_angry": [
        "That sounds frustrating. What happened to make you feel that way?",
        "It's completely normal to feel angry. Do you want to talk about it?",
        "Take a deep breath. I'm here to listen.",
    ],
    "user_feeling_tired": [
        "Sounds like you could use some rest. Make sure you take a break!",
        "You've been working hard. Maybe a little rest will help you feel refreshed.",
        "Hope you get some rest soon!",
    ],
    "user_feeling_frustrated": [
        "I'm sorry you're feeling frustrated. What can I do to help?",
        "That sounds really annoying. Take a moment to breathe.",
        "It's normal to feel that way. Let's try to figure it out together.",
    ],
    "user_feeling_worried": [
        "I'm sorry to hear you're worried. Can you tell me what's on your mind?",
        "I hope everything works out for you. I'm here to listen if you need it.",
        "It's okay to feel anxious. Just know that I'm here for you.",
    ],
    "user_feeling_confused": [
        "I can understand your confusion. What do you need help with?",
        "Sometimes things can be unclear. Let's try to clear things up.",
    ],
    "user_feeling_bored": [
        "I'm sorry you're bored. Maybe we can find something interesting to talk about?",
        "Boredom can be a tough feeling. What do you enjoy doing?",
        "Let's play a game! Or maybe you have a question I can answer?",
    ],
    "math_solve": [
        "The answer is {result}.",
        "I calculated the answer to be {result}.",
        "The result of your calculation is {result}.",
    ],
}

unknown_responses = [
    "I'm sorry, I can't understand what you said. LLW ai is a language model still developing, so there can be mistakes and errors. Please try again.",
    "My apologies, I didn't quite get that. I'm still learning, and sometimes I make mistakes. Could you please rephrase your message?",
    "It seems I'm having trouble understanding. As a developing AI, my understanding isn't perfect yet. Please try to say that again in a different way.",
]

def generate_ai_description():
    starter = random.choice(about_ai_starters)
    purpose = random.choice(about_ai_purposes)
    status = random.choice(about_ai_status)
    return f"{starter} {purpose} {status}"

def solve_math_problem(text):
    """
    Extracts and solves a math problem from a given text.
    Returns the result as a string or an error message.
    """
    # Replace text operators with symbols
    text = text.lower().replace('plus', '+')
    text = text.replace('minus', '-')
    text = text.replace('times', '*')
    text = text.replace('multiplied by', '*')
    text = text.replace('divided by', '/')
    text = text.replace('power of', '**')
    
    # Use a regex to find a mathematical expression
    match = re.search(r'[\d\s\(\)\+\-\*/\.]+', text)
    if match:
        expression = match.group(0).strip()
        try:
            # Use sympy's sympify for a safer evaluation
            result = str(sympify(expression))
            return result
        except SympifyError:
            return "I couldn't solve that math problem. Please check the expression."
    return "I couldn't find a valid math expression in your message."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_name = data.get('userName', None)

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get corrected text and uncorrected words
    corrected_text, uncorrected_words = correct_text_spelling(user_message)
    processed_message = preprocess_text(user_message)
    
    predicted_intent = model.predict([processed_message])[0]
    confidence = model.predict_proba([processed_message]).max()

    if predicted_intent == "math_solve":
        math_result = solve_math_problem(user_message)
        if "couldn't solve" in math_result:
            return jsonify({"response": math_result})
        
        response_list = intent_responses.get("math_solve", [])
        if response_list:
            ai_response_template = random.choice(response_list)
            ai_response = ai_response_template.format(result=math_result)
        else:
            ai_response = random.choice(unknown_responses)
        return jsonify({"response": ai_response})

    if predicted_intent == "ask_name" and user_name is None:
        ai_response = "I can't find your name. Please say your name so I can remember it."
        return jsonify({"response": ai_response, "action": "requestName"})

    if predicted_intent == "provide_name" and user_name is None:
        try:
            name_start_index = user_message.lower().find("my name is") + len("my name is")
            extracted_name = user_message[name_start_index:].strip().title()
            if extracted_name:
                ai_response = random.choice(intent_responses["provide_name"]).format(userName=extracted_name)
                return jsonify({"response": ai_response, "foundName": extracted_name})
        except:
            pass

    if predicted_intent == "about_ai":
        ai_response = generate_ai_description()
        return jsonify({"response": ai_response})
    
    if predicted_intent == "ask_time":
        current_time = datetime.now().strftime("%I:%M %p")
        ai_response = f"The current time is {current_time}."
        return jsonify({"response": ai_response})

    if confidence < 0.6:
        if uncorrected_words:
            # Custom response for uncorrected words
            misspelled_word = uncorrected_words[0]
            ai_response = f"I'm not sure what you mean by \"{misspelled_word}\". Could you please clarify?"
        else:
            # Fallback to generic unknown response
            ai_response = random.choice(unknown_responses)
    else:
        response_list = intent_responses.get(predicted_intent, [])
        if response_list:
            ai_response_template = random.choice(response_list)
            ai_response = ai_response_template.format(userName=user_name if user_name else 'Friend')
        else:
            ai_response = random.choice(unknown_responses)
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
