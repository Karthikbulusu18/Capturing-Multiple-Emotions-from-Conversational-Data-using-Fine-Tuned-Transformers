from collections import Counter
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout, Input, concatenate, GlobalMaxPooling1D, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset (assuming it has columns 'tweet_id', 'sentiment', and 'content')
dataset = pd.read_csv("tweet_emotions.csv")

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-\.!?,]', '', text)
    return text

# Preprocess text
dataset['text'] = dataset['content'].apply(preprocess_text)

# Define the emotion weights
emotion_weights = {
    'happy': 1.0,
    'sad': 0.7,
    'anger': 0.8,
    'fear': 0.6,
    'surprise': 0.9,
    'neutral': 0.5,
    'joy': 0.8,
    'confusion': 0.7,
    'disgust': 0.7
}

# Create a Tokenizer to index words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['text'])

X = tokenizer.texts_to_sequences(dataset['text'])
X = pad_sequences(X)

# Convert emotion labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(dataset['sentiment'])

# Inspect unique emotion labels
unique_emotions = label_encoder.classes_

# Convert emotion labels to one-hot encoded vectors
y_train_one_hot = tf.keras.utils.to_categorical(
    y_train, num_classes=len(unique_emotions))

# Split the dataset into training and testing sets
X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(
    X, y_train_one_hot, test_size=0.1, random_state=42)

# Load spaCy resources
nlp = spacy.load("en_core_web_sm")

def assign_weights(text):
    doc = nlp(text)
    weights = []
    for token in doc:
        # Assign weights based on part of speech
        if token.pos_ == 'ADJ':
            weight = 1.0  # Adjective
        elif token.pos_ == 'NOUN':
            weight = 0.7  # Noun
        elif token.pos_ == 'ADV':
            weight = 0.5  # Adverb
        elif token.pos_ == 'VERB':
            weight = 0.5  # Verb
        else:
            weight = 0.0  # Other

        # Multiply weight by emotion weight if the token is an emotion
        if token.text.lower() in emotion_weights:
            weight *= emotion_weights[token.text.lower()]

        weights.append(weight)

    return weights


# Build a CNN model
cnn_input = Input(shape=(X_train.shape[1],))
embedding_layer = Embedding(input_dim=len(
    tokenizer.word_index) + 1, output_dim=100)(cnn_input)
conv_layer = Conv1D(filters=128, kernel_size=5,
                    activation='relu')(embedding_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)

# Build a Bidirectional LSTM model
lstm_input = Input(shape=(X_train.shape[1],))
embedding_layer_lstm = Embedding(input_dim=len(
    tokenizer.word_index) + 1, output_dim=100)(lstm_input)
bi_lstm_layer = Bidirectional(
    LSTM(64, return_sequences=True))(embedding_layer_lstm)
bi_lstm_layer = Bidirectional(LSTM(64))(bi_lstm_layer)

# Combine the CNN and BiLSTM features
combined_features = concatenate([pooling_layer, bi_lstm_layer])

# Add a Dense layer for classification
dense_layer = Dense(64, activation='relu')(combined_features)
dropout_layer = Dropout(0.5)(dense_layer)

# Update the output layer to match the number of unique emotions
output_layer = Dense(len(unique_emotions), activation='softmax')(dropout_layer)

model = Model(inputs=[cnn_input, lstm_input], outputs=output_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the CNN+BiLSTM model
model.fit([X_train, X_train], y_train_one_hot, epochs=10,
          batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the CNN+BiLSTM model on test data
test_loss, test_accuracy = model.evaluate([X_test, X_test], y_test_one_hot)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Map numerical labels back to original emotion categories
predicted_labels = model.predict([X_test, X_test])
predicted_emotions = [
    unique_emotions[np.argmax(pred)] for pred in predicted_labels]
true_emotions = [unique_emotions[np.argmax(true)] for true in y_test_one_hot]

# Calculate accuracy for each emotion category
accuracy_by_emotion = {}
for emotion in unique_emotions:
    correct_predictions = [1 for pred, true in zip(
        predicted_emotions, true_emotions) if pred == emotion and true == emotion]
    total_predictions = [1 for true in true_emotions if true == emotion]
    accuracy = sum(correct_predictions) / \
        sum(total_predictions) if sum(total_predictions) > 0 else 0
    accuracy_by_emotion[emotion] = accuracy

print("Accuracy by Emotion:")
for emotion, accuracy in accuracy_by_emotion.items():
    print(f"{emotion}: {accuracy * 100:.2f}%")

# Calculate weighted accuracy for each emotion category
weighted_accuracy_by_emotion = {}
for emotion, accuracy in accuracy_by_emotion.items():
    weighted_accuracy = accuracy * emotion_weights.get(emotion, 1.0)
    weighted_accuracy_by_emotion[emotion] = weighted_accuracy

print("\nWeighted Accuracy by Emotion:")
for emotion, weighted_accuracy in weighted_accuracy_by_emotion.items():
    print(f"{emotion}: {weighted_accuracy * 100:.2f}%")
