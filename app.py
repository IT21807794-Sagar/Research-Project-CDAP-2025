from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import timedelta
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import openai
import os
import re
import extract_analyze as ea
import parse_question_response as pq
from collections import defaultdict
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import openai
import json
import spacy
from difflib import SequenceMatcher
import time
import random
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np
import re

app = Flask(__name__)

# config
app.config['SECRET_KEY'] = 'Eleav1'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config['SECRET_KEY'] = 'supersecretkey'
socketio = SocketIO(app)

# Load  modal
pipeline = joblib.load('student_clustering_pipeline.joblib')

# File to save classified students
classified_students_file = 'classified_students.csv'

# Stress Detection Setup
stress_model = load_model("stress_detection_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
stress_count = 0
start_time = time.time()
stress_level = "-"
cap = cv2.VideoCapture(0)


# home route
@app.route("/home")
def home():
    return render_template("home.html")


# classify  level by stress
def classify_stress_level(stress_count):
    if stress_count < 50:
        return "no_stress"
    elif 50 <= stress_count < 110:
        return "low"
    elif 110 <= stress_count < 230:
        return "moderate"
    else:
        return "high"


# live video frames
def generate_frames():
    global stress_count, start_time, stress_level
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))
            face_array = np.expand_dims(face_resized, axis=0) / 255.0
            prediction = stress_model.predict(face_array)
            label = "Stress" if prediction > 0.5 else "No Stress"
            if prediction > 0.5:
                stress_count += 1
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 30:
            stress_level = classify_stress_level(stress_count)
            stress_count = 0
            start_time = time.time()

        cv2.putText(frame, f"Stress Level: {stress_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Stress Frames (30s): {stress_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ChatBot modela
lemmatizer = WordNetLemmatizer()
chat_model = load_model('ChatBot/chatbot.h5')
words = pickle.load(open('ChatBot/words.pkl', 'rb'))
classes = pickle.load(open('ChatBot/classes.pkl', 'rb'))
intents = json.loads(open("ChatBot/Data.json").read())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    bag = pad_sequences([bag], maxlen=len(words), padding='post')[0]
    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words)
    res = chat_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(ints):
    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I don't understand."


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    ints = predict_class(user_message)
    response = get_response(ints) if ints else "I didn't get that. Can you rephrase?"
    return jsonify({"response": response})


# basic Routes
@app.route('/')
def Emotional_Awareness_index():
    return render_template('Emotional_Awareness_index.html', stress_level=stress_level)


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_stress_level')
def get_stress_level():
    global stress_level
    return jsonify({"stress_level": stress_level})


# 2---------------------------------------------------------------------------------------------------**********-------------


#  Helper Function
def get_level(cluster):
    """
    Map cluster number to Low / Medium / High level
    Adjust based on your clusters
    """
    if cluster in [0, 1]:
        return "Low Level"
    elif cluster in [2, 3, 4]:
        return "Medium Level"
    else:
        return "High Level"


# basic Routes
@app.route('/GC_index', methods=['GET', 'POST'])
def GC_index():
    results = None
    if request.method == 'POST':
        uploaded_file = request.files['csv']
        if uploaded_file.filename != '':
            df = pd.read_csv(uploaded_file)

            # Predict clusters
            df['cluster'] = pipeline.predict(df)

            # Map clusters to levels
            df['level'] = df['cluster'].apply(get_level)

            if 'id' not in df.columns:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'id'}, inplace=True)

            # Save results
            df.to_csv(classified_students_file, index=False)
            results = df.to_dict(orient='records')

    return render_template('GC_index.html', results=results, filename=classified_students_file)


@app.route('/group_chat')
def group_chat():
    df = pd.read_csv(classified_students_file)

    # Map clusters
    df['level'] = df['cluster'].apply(get_level)

    if 'id' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)

    # Separate students
    low_students = df[df['level'] == 'Low Level'].to_dict(orient='records')
    medium_students = df[df['level'] == 'Medium Level'].to_dict(orient='records')
    high_students = df[df['level'] == 'High Level'].to_dict(orient='records')

    # Form mixed groups
    num_groups = min(len(low_students), len(medium_students), len(high_students))
    chats = []
    for i in range(num_groups):
        group = [low_students[i], medium_students[i], high_students[i]]
        chats.append(group)

    return render_template('group_chat_live.html', chats=chats)


@socketio.on('join')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    emit('status', {'msg': f"{username} has joined the chat."}, room=room)


@socketio.on('text')
def on_text(data):
    room = data['room']
    msg = data['msg']
    username = data['username']
    emit('message', {'msg': f"{username}: {msg}"}, room=room)


@socketio.on('leave')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    emit('status', {'msg': f"{username} has left the chat."}, room=room)


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


