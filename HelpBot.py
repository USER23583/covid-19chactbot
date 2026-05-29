import json
import random
import numpy as np
import re
import os
import pickle

from flask import Flask, render_template, request, jsonify

# TensorFlow Imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input 

# -----------------------------
# Stop Words
# -----------------------------
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but',
    'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'is', 'are', 'was', 'were'
}

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')


class CoronaChatBot:
    def __init__(self, json_path):
        self.json_path = json_path
        self.dialogue_info = None
        self.unique_vocab = []
        self.intent_classes = []
        self.feature_matrices = []
        self.target_vectors = []
        self.trained_model = None
        self.last_topic = None

        self.load_and_process_data()
        self.setup_model()

    # -----------------------------
    # Clean Text
    # -----------------------------
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = text.split()

        cleaned_words = [
            word for word in words
            if word not in STOP_WORDS and len(word) > 1
        ]

        return cleaned_words

    # -----------------------------
    # Load and Process Data
    # -----------------------------
    def load_and_process_data(self):
        cache_name = "processed_dialogue.pkl"

        with open(self.json_path, "r", encoding="utf-8") as file:
            self.dialogue_info = json.load(file)

        json_mtime = os.path.getmtime(self.json_path)
        cache_is_valid = False

        if os.path.exists(cache_name):
            cache_mtime = os.path.getmtime(cache_name)
            cache_is_valid = cache_mtime >= json_mtime

        # Load cache if exists and is up-to-date
        if cache_is_valid:
            with open(cache_name, "rb") as cache:
                (
                    self.unique_vocab,
                    self.intent_classes,
                    self.feature_matrices,
                    self.target_vectors
                ) = pickle.load(cache)

            print("Processed data loaded from cache")
            return

        all_words = []
        documents = []

        for intent in self.dialogue_info["intents"]:
            tag = intent["tag"]

            if tag not in self.intent_classes:
                self.intent_classes.append(tag)

            for pattern in intent["patterns"]:
                words = self.clean_text(pattern)

                all_words.extend(words)
                documents.append((words, tag))

        self.unique_vocab = sorted(list(set(all_words)))
        self.intent_classes = sorted(self.intent_classes)

        training_data = []
        output_empty = [0] * len(self.intent_classes)

        for doc in documents:
            bag = []
            pattern_words = doc[0]

            for word in self.unique_vocab:
                bag.append(1 if word in pattern_words else 0)

            output_row = output_empty[:]
            output_row[self.intent_classes.index(doc[1])] = 1

            training_data.append([bag, output_row])

        random.shuffle(training_data)

        training_data = np.array(training_data, dtype=object)

        self.feature_matrices = np.array(list(training_data[:, 0]), dtype=np.float32)
        self.target_vectors = np.array(list(training_data[:, 1]), dtype=np.float32)

        # Save cache
        with open(cache_name, "wb") as cache:
            pickle.dump(
                (
                    self.unique_vocab,
                    self.intent_classes,
                    self.feature_matrices,
                    self.target_vectors
                ),
                cache
            )

        print("Data processed successfully")

    # -----------------------------
    # Setup Model
    # -----------------------------
    def setup_model(self):
        model_name = "corona_bot_model.keras"
        json_mtime = os.path.getmtime(self.json_path)

        # Load model if it exists and is newer than the intent file
        if os.path.exists(model_name):
            model_mtime = os.path.getmtime(model_name)
            if model_mtime >= json_mtime:
                self.trained_model = load_model(model_name)
                print("Existing model loaded")
                return

        # Build model
        self.trained_model = Sequential([
            Input(shape=(len(self.feature_matrices[0]),)),

            Dense(128, activation="relu"),
            Dropout(0.4),

            Dense(64, activation="relu"),
            Dropout(0.3),

            Dense(len(self.target_vectors[0]), activation="softmax")
        ])

        # Compile model
        self.trained_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train model
        self.trained_model.fit(
            self.feature_matrices,
            self.target_vectors,
            epochs=200,
            batch_size=8,
            validation_split=0.1,
            verbose=1
        )

        # Save model
        self.trained_model.save(model_name)

        print("Model trained and saved successfully")

    # -----------------------------
    # Create Feature Vector
    # -----------------------------
    def create_feature_vector(self, sentence):
        cleaned_input = self.clean_text(sentence)

        bag = [
            1 if word in cleaned_input else 0
            for word in self.unique_vocab
        ]

        return np.array([bag], dtype=np.float32)

    # -----------------------------
    # Generate Response
    # -----------------------------
    def get_response(self, user_input):
        if not user_input.strip():
            return "Please enter a message."

        input_vector = self.create_feature_vector(user_input)

        predictions = self.trained_model.predict(input_vector, verbose=0)[0]

        max_index = np.argmax(predictions)
        confidence = float(predictions[max_index])
        predicted_tag = self.intent_classes[max_index]

        if confidence > 0.25:
            for intent in self.dialogue_info["intents"]:
                if intent["tag"] == predicted_tag:
                    return random.choice(intent["responses"])

        return "Sorry, I could not understand your message. Please ask in a simpler way."


# -----------------------------
# Initialize ChatBot
# -----------------------------
chatbot = CoronaChatBot("who.json")


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_message = data.get("message", "")

    response = chatbot.get_response(user_message)

    return jsonify({
        "response": response
    })


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)