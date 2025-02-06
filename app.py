from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Models
lang_detector = joblib.load('language_detector.joblib')
lang_vectorizer = joblib.load('lang_vectorizer.joblib')
models = {
    'en': joblib.load('en_intent.joblib'),
    'es': joblib.load('es_intent.joblib'),
    'fr': joblib.load('fr_intent.joblib'),
}

# Culturally Appropriate Responses
RESPONSES = {
    'en': {
        'greet': ['Hello!', 'Hi there!', 'How can I help you?'],
        'book': ['Booking your flight...', 'I’ll check available flights.'],
        'weather': ['It’s sunny today!', 'Rain expected tomorrow.']
    },
    'es': {
        'greet': ['¡Hola!', '¿En qué puedo ayudarte?'],
        'book': ['Reservando tu vuelo...', 'Verificando vuelos disponibles.'],
        'weather': ['¡Hoy hace sol!', 'Se espera lluvia mañana.']
    },
    'fr': {
        'greet': ['Bonjour !', 'Comment puis-je vous aider ?'],
        'book': ['Réservation de votre vol...', 'Je vérifie les vols disponibles.'],
        'weather': ['Il fait beau aujourd’hui !', 'Pluie prévue demain.']
    },
    'de': {
        'greet': ['Hallo!', 'Wie kann ich helfen?'],
        'book': ['Buche deinen Flug...', 'Ich prüfe verfügbare Flüge.'],
        'weather': ['Heute ist es sonnig!', 'Morgen wird Regen erwartet.']
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        # Language Detection
        lang = lang_detector.predict(lang_vectorizer.transform([text]))[0]
        # Intent Classification
        tfidf, model = models[lang]
        intent = model.predict(tfidf.transform([text]))[0]
        # Generate Response
        response = np.random.choice(RESPONSES[lang][intent])
        return jsonify({'response': response, 'lang': lang})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)