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


###3- PR---------------------------------------------------------------****--------------------------------


# Track  questions
def init_question_tracking():
    if "generated_questions" not in session:
        session["generated_questions"] = []


def add_generated_question(question_text):
    if "generated_questions" not in session:
        session["generated_questions"] = []

    #  hash the question
    question_hash = hash(question_text.lower().strip())
    if question_hash not in session["generated_questions"]:
        session["generated_questions"].append(question_hash)
        session.modified = True


def is_question_repeated(question_text):
    if "generated_questions" not in session:
        return False

    question_hash = hash(question_text.lower().strip())
    return question_hash in session["generated_questions"]


#  user performance
def init_performance_tracking():
    if "performance" not in session:
        session["performance"] = {
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_topics": {},
            "current_streak": 0
        }
    elif isinstance(session["performance"].get("incorrect_topics"), defaultdict):
        session["performance"]["incorrect_topics"] = dict(session["performance"]["incorrect_topics"])


def update_incorrect_topic(topic_name):
    if "incorrect_topics" not in session["performance"]:
        session["performance"]["incorrect_topics"] = {}

    safe_key = re.sub(r'[^a-zA-Z0-9]', '_', topic_name.lower())[:30]

    if safe_key in session["performance"]["incorrect_topics"]:
        session["performance"]["incorrect_topics"][safe_key]["count"] += 1
    else:
        session["performance"]["incorrect_topics"][safe_key] = {
            "name": topic_name,
            "count": 1
        }
    session.modified = True


def get_incorrect_topics():
    incorrect_data = session["performance"].get("incorrect_topics", {})
    return {data["name"]: data["count"] for data in incorrect_data.values()}


# Analyze document
def analyze_with_openai(text_content):
    try:
        content_sample = text_content[:6000]

        prompt = f"""
        Analyze the following text content from a document and identify 3-5 specific, 
        distinct main topics or subjects that would be appropriate for creating 
        short-answer questions. Focus on the most prominent and unique themes.

        Return ONLY a numbered list (1. Topic Name) without any additional commentary.

        Text content:
        {content_sample}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        raw_text = response.choices[0].message.get("content", "")

        topics = {}
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        for i, line in enumerate(lines, 1):
            if re.match(r'^\d+\.', line):
                parts = line.split('.', 1)
                if len(parts) == 2:
                    topic = parts[1].strip()
                    topics[str(i)] = topic
            elif i <= 5:
                topics[str(i)] = line

        return topics if topics else {
            "1": "Main Concepts",
            "2": "Key Themes",
            "3": "Important Topics",
            "4": "Core Ideas",
            "5": "Primary Subjects"
        }

    except Exception as e:
        print(f"Analysis error: {e}")
        return {
            "1": "Main Concepts",
            "2": "Key Themes",
            "3": "Important Topics",
            "4": "Core Ideas",
            "5": "Primary Subjects"
        }


# Generate questions
def generate_question(subject, difficulty="medium", max_attempts=3):
    for attempt in range(max_attempts):
        prompt = f"""
        Create a {difficulty} difficulty short-answer question specifically about: {subject}

        Important Guidelines for Primary Level Students:
        1. Use mathematical symbols (+, -, ×, ÷, =) instead of words when appropriate
        2. Questions should be simple, clear and age-appropriate
        3. For math questions, use symbols like: 5 + 3 = ? instead of "five plus three"
        4. Make questions engaging and educational for young students

        Requirements:
        1. Question should be clear, specific, and directly related to {subject}
        2. Correct answer should be concise but comprehensive
        3. Explanation should be brief but informative
        4. Make the question unique and tailored to this specific subject

        Format exactly as:
        Question: [question text]
        Correct: [correct answer]
        Explanation: [explanation text]
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300
            )

            raw_text = response.choices[0].message.get("content", "")
            question_data = pq.parse_question_response(raw_text.strip())

            if question_data and not is_question_repeated(question_data["question"]):
                add_generated_question(question_data["question"])
                return question_data

        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")

    return None


#  answer evaluation
def check_answer(question_data, user_answer):
    correct_answer = question_data["correct"].lower().strip()
    user_answer_clean = user_answer.lower().strip()

    # filler words
    filler_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'please', 'answer', 'result']
    user_words = [word for word in user_answer_clean.split() if word not in filler_words]
    user_answer_clean = ' '.join(user_words)

    #  mathematical expressions and symbols
    if any(op in question_data["question"] for op in ['+', '-', '×', '÷', '*', '/', '=']):
        return evaluate_math_answer(question_data, user_answer_clean)

    #  number word conversions
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15'
    }

    # Convert  answers
    for word, digit in number_words.items():
        correct_answer = correct_answer.replace(word, digit)
        user_answer_clean = user_answer_clean.replace(word, digit)

    # Remove units
    correct_without_units = re.sub(r'\b(apples?|oranges?|items?|objects?|units?)\b', '', correct_answer).strip()
    user_without_units = re.sub(r'\b(apples?|oranges?|items?|objects?|units?)\b', '', user_answer_clean).strip()

    # Multiple comparison
    comparison_methods = [
        lambda: user_answer_clean == correct_answer,
        lambda: user_without_units == correct_without_units,
        lambda: extract_numbers(user_answer_clean) == extract_numbers(correct_answer) and extract_numbers(
            correct_answer),
        lambda: calculate_similarity(user_answer_clean, correct_answer) > 0.7
    ]

    for method in comparison_methods:
        try:
            if method():
                return True, question_data["correct"], question_data["explanation"]
        except:
            continue

    return False, question_data["correct"], question_data["explanation"]


def evaluate_math_answer(question_data, user_answer):
    """Special evaluation for mathematical questions"""
    correct_answer = question_data["correct"].lower().strip()
    user_answer_clean = user_answer.lower().strip()

    # Extract numerical values
    correct_numbers = extract_numbers(correct_answer)
    user_numbers = extract_numbers(user_answer_clean)

    if correct_numbers and user_numbers:
        if any(num in user_numbers for num in correct_numbers):
            return True, question_data["correct"], question_data["explanation"]

    try:
        # Extract the mathematical expression
        question = question_data["question"]
        if any(op in question for op in ['+', '-', '×', '÷', '*', '/']):
            result = evaluate_simple_math(question)
            if result and str(result) in user_answer_clean:
                return True, question_data["correct"], question_data["explanation"]
    except:
        pass

    return check_answer_fallback(question_data, user_answer)


def extract_numbers(text):
    """Extract all numbers from text, including words"""
    # number words to digits
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }

    for word, digit in number_words.items():
        text = text.replace(word, digit)

    # Extract values
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(num) if '.' in num else int(num) for num in numbers]


def evaluate_simple_math(question):
    """Evaluate simple mathematical expressions"""
    try:
        # Extract the mathematic
        math_expr = re.search(r'(\d+\s*[\+\-\*\/\×\÷]\s*\d+)', question)
        if math_expr:
            expr = math_expr.group(1)
            expr = expr.replace('×', '*').replace('÷', '/')
            result = eval(expr)
            return result
    except:
        pass
    return None


def calculate_similarity(text1, text2):
    """Calculate simple text similarity for primary level answers"""
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def check_answer_fallback(question_data, user_answer):
    """Fallback answer checking method"""
    correct = question_data["correct"].lower().strip()
    user_answer_clean = user_answer.lower().strip()

    correct_terms = set(correct.split())
    user_terms = set(user_answer_clean.split())

    if not correct_terms:
        return False, question_data["correct"], question_data["explanation"]

    match_score = len(correct_terms & user_terms) / len(correct_terms)

    # Lower threshold
    if match_score > 0.4:
        return True, question_data["correct"], question_data["explanation"]
    else:
        return False, question_data["correct"], question_data["explanation"]


# Generate revision
def generate_revision_tips(incorrect_topics):
    if not incorrect_topics:
        return "Great job! You've answered all questions correctly. Keep up the good work!"

    topics_list = ", ".join([f"{topic} ({count} wrong)" for topic, count in incorrect_topics.items()])

    prompt = f"""
    Based on the following topics that a student has struggled with, provide 2-3 concise revision tips.
    The student has answered questions incorrectly on: {topics_list}

    Provide specific study suggestions for these areas in a friendly, encouraging tone.
    Format as a short paragraph followed by bullet points.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250
        )

        return response.choices[0].message.get("content", "")
    except Exception as e:
        print(f"Revision tips error: {e}")
        return "Focus on reviewing the topics you've answered incorrectly. Practice makes perfect!"


# Question history
def init_question_history():
    if "question_history" not in session:
        session["question_history"] = []


def add_to_question_history(question_data, user_answer, is_correct, topic):
    history_entry = {
        "question": question_data["question"],
        "user_answer": user_answer,
        "correct_answer": question_data["correct"],
        "explanation": question_data["explanation"],
        "is_correct": is_correct,
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "difficulty": session.get("difficulty", "medium")
    }

    session["question_history"].append(history_entry)
    if len(session["question_history"]) > 50:
        session["question_history"] = session["question_history"][-50:]
    session.modified = True


def generate_performance_graph():
    try:
        history = session.get("question_history", [])

        if len(history) < 2:
            return None

        daily_data = {}
        for entry in history:
            date = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d')
            if date not in daily_data:
                daily_data[date] = {'total': 0, 'correct': 0}

            daily_data[date]['total'] += 1
            if entry['is_correct']:
                daily_data[date]['correct'] += 1

        dates = sorted(daily_data.keys())
        scores = [(daily_data[date]['correct'] / daily_data[date]['total'] * 100) for date in dates]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, scores, marker='o', linewidth=2, markersize=8, color='#4e73df')
        plt.fill_between(dates, scores, alpha=0.3, color='#4e73df')
        plt.title('Performance Trend Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Score (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        for i, (date, score) in enumerate(zip(dates, scores)):
            plt.annotate(f'{score:.0f}%', (date, score),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontweight='bold')

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{graphic}"

    except Exception as e:
        print(f"Graph generation error: {e}")
        return None


# Routes
@app.route("/PR_index", methods=["GET", "POST"])
def PR_index():
    if request.method == "POST":
        session.clear()
        init_performance_tracking()
        init_question_tracking()

        file = request.files.get("pdf")
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            text_content = ea.extract_content_from_pdf(filepath)

            try:
                os.remove(filepath)
            except:
                pass

            topics = analyze_with_openai(text_content)
            if topics:
                session["topics"] = topics
                return redirect(url_for("topics"))
            else:
                return render_template("PR_index.html", error="Failed to analyze the document")
        else:
            return render_template("PR_index.html", error="Please upload a valid PDF file")

    return render_template("PR_index.html")


@app.route("/topics", methods=["GET", "POST"])
def topics():
    topics = session.get("topics", {})

    if not topics:
        return redirect(url_for("index"))

    if request.method == "POST":
        selected_topic = request.form["topic"]

        for key, value in session.items():
            print(f"{key}: {value}")
        print("==============================")

        # Get skill level
        skill_level = session.get("skill_level")

        if not skill_level and 'performance_data' in session:
            print("Recalculating skill level from performance data...")
            skill_level = classify_student(session['performance_data'])
            session['skill_level'] = skill_level
            session.modified = True

        if not skill_level:
            skill_level = "Moderately Skilled"

        # Map skill_level
        if skill_level == "Highly Skilled":
            difficulty = "hard"
        elif skill_level == "Moderately Skilled":
            difficulty = "medium"
        else:
            difficulty = "easy"

        print(f"Final decision - Skill: {skill_level}, Difficulty: {difficulty}")

        session["selected_topic"] = selected_topic
        session["difficulty"] = difficulty
        session.pop("question_data", None)
        session["question_set_count"] = 0

        return redirect(url_for("question"))

    skill_level = session.get("skill_level", "Not assessed")

    return render_template("topics.html", topics=topics, skill_level=skill_level)


@app.route("/question", methods=["GET", "POST"])
def question():
    selected_topic = session.get("selected_topic")
    difficulty = session.get("difficulty")

    init_performance_tracking()
    init_question_history()
    init_question_tracking()

    if not selected_topic:
        return redirect(url_for("index"))

    if "question_set_count" not in session:
        session["question_set_count"] = 0

    if "question_data" not in session:
        question_data = generate_question(selected_topic, difficulty)
        if question_data:
            session["question_data"] = question_data
            session["question_set_count"] = session.get("question_set_count", 0) + 1
        else:
            return render_template("error.html", message="Failed to generate question. Please try a different topic.")

    question_data = session["question_data"]
    question_number = session.get("question_set_count", 1)
    total_questions = session["performance"].get("total_questions", 0)

    if request.method == "POST":
        user_answer = request.form["answer"]
        is_correct, correct, explanation = check_answer(question_data, user_answer)

        session["performance"]["total_questions"] = session["performance"].get("total_questions", 0) + 1
        if is_correct:
            session["performance"]["correct_answers"] = session["performance"].get("correct_answers", 0) + 1
            session["performance"]["current_streak"] = session["performance"].get("current_streak", 0) + 1
        else:
            session["performance"]["current_streak"] = 0
            update_incorrect_topic(selected_topic)

        add_to_question_history(question_data, user_answer, is_correct, selected_topic)

        if session["question_set_count"] % 5 == 0:
            return redirect(url_for("revision"))

        session.pop("question_data", None)

        return render_template(
            "result.html",
            question=question_data["question"],
            user_answer=user_answer,
            correct=is_correct,
            expected=correct,
            explanation=explanation,
            question_number=question_number,
            total_questions=total_questions + 1,
            performance=session["performance"]
        )

    return render_template(
        "question.html",
        question=question_data["question"],
        question_number=question_number,
        total_questions=total_questions,
        difficulty=difficulty.capitalize(),
        performance=session["performance"]
    )


@app.route("/revision")
def revision():
    performance = session.get("performance", {})
    total = performance.get("total_questions", 0)
    correct = performance.get("correct_answers", 0)
    score = (correct / total * 100) if total > 0 else 0

    incorrect_topics = get_incorrect_topics()

    if incorrect_topics:
        tips = generate_revision_tips(incorrect_topics)
    else:
        tips = None

    return render_template(
        "revision.html",
        score=score,
        total_questions=total,
        correct_answers=correct,
        revision_tips=tips,
        incorrect_topics=incorrect_topics
    )


@app.route("/reset_question_set")
def reset_question_set():
    session["question_set_count"] = 0
    session.pop("question_data", None)
    # Clear generated questions
    if "generated_questions" in session:
        session["generated_questions"] = []
    return redirect(url_for("question"))


@app.route("/new_question")
def new_question():
    session.pop("question_data", None)
    return redirect(url_for("question"))


@app.route("/evaluate", methods=["POST"])
def evaluate():
    user_answer = request.form.get("answer", "")
    question_data = session.get("question_data")

    if not question_data:
        return jsonify({"is_correct": False, "feedback": "No active question.", "correct_answer": ""})

    is_correct, correct, explanation = check_answer(question_data, user_answer)

    return jsonify({
        "is_correct": is_correct,
        "feedback": "Great job! " if is_correct else "Not quite right. Try again!",
        "correct_answer": correct,
        "explanation": explanation
    })


@app.route("/question_history")
def question_history():
    history = session.get("question_history", [])
    return jsonify(history)


@app.route("/performance_data")
def performance_data():
    graph_data = generate_performance_graph()
    return jsonify({"graph": graph_data})


@app.route("/restart")
def restart():
    session.clear()
    return redirect(url_for("index"))

