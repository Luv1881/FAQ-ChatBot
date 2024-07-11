from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import csv
import nltk
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__, static_url_path='', static_folder='static')
app.secret_key = 'supersecretkey'
CORS(app)

logging.basicConfig(level=logging.INFO)

faqs = []
with open('faq.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row['question'].strip()
        answer = row['answer'].strip()
        faqs.append((question, answer))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def get_named_entities(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = " ".join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))
    return entities

processed_faq_questions = [preprocess_text(question) for question, answer in faqs]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_faq_questions)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def answer_faq_bert(question):
    inputs = tokenizer(question, return_tensors='pt')
    answer_texts = []
    for faq_question, faq_answer in faqs:
        context = faq_question
        inputs.update(tokenizer(context, return_tensors='pt'))
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
        answer_texts.append((answer, faq_answer))
    best_answer = max(answer_texts, key=lambda x: len(x[0]))
    if len(best_answer[0]) > 0:
        return best_answer[1]
    else:
        return "I'm sorry, I don't understand the question."

def answer_faq(question):
    processed_question = preprocess_text(question)
    question_tfidf = vectorizer.transform([processed_question])
    cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
    best_match_index = np.argmax(cosine_similarities)
    best_match_score = cosine_similarities[best_match_index]

    if best_match_score > 0.1:
        return faqs[best_match_index][1]
    else:
        return answer_faq_bert(question)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_input = data.get('question', '')
        sentiment = sia.polarity_scores(user_input)
        entities = get_named_entities(user_input)
        response = answer_faq(user_input)
        session.setdefault('chat_history', []).append({'question': user_input, 'answer': response})
        return jsonify({'answer': response, 'sentiment': sentiment, 'entities': entities})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'answer': "An error occurred. Please try again later."}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        question = data.get('question', '')
        feedback = data.get('feedback', '')
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
