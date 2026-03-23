import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
data = pd.read_csv("dataset.csv")

texts = data['text']
labels = data['disease']

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Tokenize text
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded = pad_sequences(sequences, maxlen=10)

# Build model
model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=16, input_length=10))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(len(set(labels_encoded)), activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(padded, labels_encoded, epochs=50)

# Save model
model.save("model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model trained and saved successfully!")