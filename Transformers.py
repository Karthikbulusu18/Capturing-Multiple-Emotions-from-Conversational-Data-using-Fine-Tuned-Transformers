import pandas as pd
import numpy as np
import spacy
import re
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam

# Load the MELD dataset
dataset = pd.read_csv("tweet_emotions.csv")

# Preprocessing function for MELD dataset


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-\.!?,]', '', text)
    return text


# Update this line to match your dataset column name
dataset['text'] = dataset['Utterance'].apply(preprocess_text)

# Load spaCy resources
nlp = spacy.load("en_core_web_sm")

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

# Convert emotion labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['Emotion'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset['text'], dataset['label'], test_size=0.2, random_state=42)

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

dataset['weights'] = dataset['text'].apply(assign_weights)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(label_encoder.classes_))

# Encode the data using BERT tokenizer
train_encodings = tokenizer(
    list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(
    list(X_test), truncation=True, padding=True, max_length=128)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(len(X_train)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(64)

# Define optimizer and loss function
optimizer = Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the BERT model
model.fit(train_dataset, epochs=25)

# Evaluate the BERT model on test data
results = model.evaluate(test_dataset)
print("Test loss, Test accuracy:", results)

# Predict emotions on the test data
y_pred = model.predict(test_dataset)
y_pred_labels = np.argmax(y_pred.logits, axis=1)

# Map numerical labels back to original emotion categories
predicted_emotions = label_encoder.inverse_transform(y_pred_labels)
true_emotions = label_encoder.inverse_transform(y_test)

# Calculate weighted accuracy for each emotion category
accuracy_by_emotion = {}
for emotion in np.unique(true_emotions):
    correct_predictions = np.sum(
        [1 for pred, true in zip(predicted_emotions, true_emotions) if pred == emotion and true == emotion])
    total_predictions = np.sum(
        [1 for true in true_emotions if true == emotion])
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    accuracy_weighted = accuracy * emotion_weights.get(emotion, 1.0)
    accuracy_by_emotion[emotion] = accuracy_weighted

print("Weighted Accuracy by Emotion:")
for emotion, accuracy in accuracy_by_emotion.items():
    print(f"{emotion}: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(true_emotions, predicted_emotions))
