from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import threading
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
stress_count = 0
start_time = time.time()
stress_level = "-"
model = None
face_cascade = None
cap = None
is_running = False

# chatbot Confi
lemmatizer = WordNetLemmatizer()
chatbot_model = None
chatbot_words = None
chatbot_classes = None
chatbot_intents = None

def initialize_stress_detection():
    global model, face_cascade
    #  stress  model
    model = load_model("stress_detection_model.h5")
    #  Haar Cascade 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def initialize_chatbot():
    global chatbot_model, chatbot_words, chatbot_classes, chatbot_intents
    # trained model 
    chatbot_model = load_model('ChatBot/chatbot.h5')
    chatbot_words = pickle.load(open('ChatBot/words.pkl', 'rb'))
    chatbot_classes = pickle.load(open('ChatBot/classes.pkl', 'rb'))
    chatbot_intents = json.loads(open("ChatBot/Data.json").read())

def classify_stress_level(stress_count):
    if stress_count < 50:
        return "no_stress"
    elif 50 <= stress_count < 110:
        return "low"
    elif 110 <= stress_count < 230:
        return "moderate"
    else:  # stress_count >= 230
        return "high"

def generate_frames():
    global stress_count, start_time, stress_level, is_running
    
    cap = cv2.VideoCapture(0)
    is_running = True
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128, 128))
            face_array = np.expand_dims(face_resized, axis=0)
            face_array = face_array / 255.0 

            prediction = model.predict(face_array)
            label = "Stress" if prediction > 0.5 else "No Stress"

            if prediction > 0.5:
                stress_count += 1

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 30:
            stress_level = classify_stress_level(stress_count)
            start_time = time.time()
            stress_count = 0

        cv2.putText(frame, f"Stress Level: {stress_level.replace('_', ' ').capitalize()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Stress Frames (30s): {stress_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# Chatbot 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    bag = pad_sequences([bag], maxlen=len(words), padding='post')[0]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, chatbot_words, show_details=False)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": chatbot_classes[r[0]], "probability": str(r[1])})
    return return_list

def get_chatbot_response(ints):
    tag = ints[0]['intent']
    list_of_intents = chatbot_intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stress_detection', methods=['POST'])
def start_stress_detection():
    global is_running
    if not is_running:
        initialize_stress_detection()
        threading.Thread(target=generate_frames).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/stop_stress_detection', methods=['POST'])
def stop_stress_detection():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/get_stress_level', methods=['GET'])
def get_stress_level():
    return jsonify({
        "stress_level": stress_level,
        "stress_count": stress_count
    })

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message', '')
    ints = predict_class(message)
    response = get_chatbot_response(ints)
    return jsonify({"response": response})

if __name__ == '__main__':
    initialize_chatbot()
    app.run(debug=True, threaded=True)