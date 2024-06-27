from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import csv
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

faqs = []
with open('faq.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row['question'].strip()
        answer = row['answer'].strip()
        faqs.append((question, answer))

def answer_faq(question):
    question_tokens = word_tokenize(question.lower())
    best_match_answer = "I'm sorry, I don't understand the question."
    best_match_count = 0

    for faq_question, faq_answer in faqs:
        faq_question_tokens = word_tokenize(faq_question.lower())
        match_count = sum(1 for word in faq_question_tokens if word in question_tokens)
        if match_count > best_match_count:
            best_match_count = match_count
            best_match_answer = faq_answer

    return best_match_answer

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get('question', '')
    response = answer_faq(user_input)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
