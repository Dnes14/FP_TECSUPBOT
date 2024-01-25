import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz

# Ingreso

lemmatizer = WordNetLemmatizer()

# Cargar datos de intents
intents = json.loads(open('intents.json').read())

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Procesar palabras y clases
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']


#Proceso

#Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Eliminar duplicados y ordenar palabras
words = sorted(set(lemmatizer.lemmatize(word) for word in words if word not in ignore_letters))
classes = sorted(set(classes))

# Guardar palabras y clases
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento

# Crear bolsas y etiquetas
training = []
outputEmpty = [0] * len(classes)

max_bag_length = max(len(item[0]) for item in documents)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
print(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Crear modelo de red neuronal
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar y guardar modelo

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

#salida

print('Done')
