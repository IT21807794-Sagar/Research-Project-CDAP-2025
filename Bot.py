import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import random

# Load the trained model and necessary files
lemmatizer = WordNetLemmatizer()
model = load_model('ChatBot/chatbot.h5')
words = pickle.load(open('ChatBot/words.pkl', 'rb'))
classes = pickle.load(open('ChatBot/classes.pkl', 'rb'))
intents = json.loads(open("ChatBot/Data.json").read())

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
    # Pad the bag to match the expected input length
    bag = pad_sequences([bag], maxlen=len(words), padding='post')[0]
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

# Terminal-based chat
print("Bot: Hello! I'm your stress management assistant. How can I help you today? (Type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Goodbye! Take care of yourself.")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)