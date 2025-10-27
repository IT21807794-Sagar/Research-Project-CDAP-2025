import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

# WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# corpus pre-processing
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("ChatBot/Data.json").read()
intents = json.loads(data_file)

# tokenization
nltk.download('punkt')
nltk.download('wordnet')
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Creating a pickle
pickle.dump(words, open('ChatBot/words.pkl', 'wb'))
pickle.dump(classes, open('ChatBot/classes.pkl', 'wb'))
training = []
output_empty = [0] * len(classes)

# Training set
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# convert to into NumPy arrays
random.shuffle(training)

# Separate features (X) and labels (Y)
X = np.array([sample[0] for sample in training])
Y = np.array([sample[1] for sample in training])
X = pad_sequences(X, maxlen=len(words), padding='post')

# Create NN model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(X, Y, epochs=100, batch_size=5, verbose=1)
model.save('ChatBot/chatbot.h5', hist)
print("\n")
print("*" * 50)
print("\nModel Created Successfully!")
