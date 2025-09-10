from flask import Flask, render_template, request, jsonify
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Mistral API setup
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    chat_history = request.json.get('history', [])

    # Add the user's message to the chat history
    chat_history.append({"role": "user", "content": user_message})

    # Mistral streaming response
    stream_response = client.chat.stream(
        model=model,
        messages=chat_history
    )

    response_content = ""
    for chunk in stream_response:
        response_content += chunk.data.choices[0].delta.content

    # Add the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": response_content})

    return jsonify({"response": response_content, "history": chat_history})

if __name__ == '__main__':
    app.run(debug=True)
