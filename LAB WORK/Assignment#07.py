print("\n--- Q1: Next Word Prediction using RNN ---\n")
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

data = """The sun is shining. The sky is blue. The sun is bright. The day is beautiful."""

corpus = data.lower().replace('.', '').split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(SimpleRNN(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

def predict_next_word(seed_text, n_words=1):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0))
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

print(predict_next_word("the sun is"))

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q2: Stock Price Prediction using RNN. ---\n")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

df = yf.download('GOOG', start='2015-01-01', end='2020-01-01')
training_set = df[['Open']].values

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

X_train = []
y_train = []
time_step = 60
for i in range(time_step, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-time_step:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(SimpleRNN(units=80, activation='tanh', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

predicted_stock_price = model.predict(X_train)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
real_stock_price = scaler.inverse_transform(training_set_scaled)

plt.figure(figsize=(10,6))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction using RNN')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q3: Sentiment Analysis using RNN. ---\n")
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

model = Sequential()
model.add(Embedding(10000, 32, input_length=200))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", acc)

sample = pad_sequences([[1,14,20,43,2,15]], maxlen=200)
print("Prediction:", model.predict(sample))

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q4: Weather Forecasting using RNN. ---\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

data = pd.read_csv("DataFiles\jena_climate_2009_2016.csv")
temps = data['T (degC)'].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(temps)

X, y = [], []
time_step = 30
for i in range(time_step, len(scaled)):
    X.append(scaled[i-time_step:i,0])
    y.append(scaled[i,0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(SimpleRNN(64, input_shape=(X.shape[1],1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

pred = model.predict(X)
plt.plot(scaler.inverse_transform(y.reshape(-1,1)), color='blue', label='Actual Temperature')
plt.plot(scaler.inverse_transform(pred), color='red', label='Predicted Temperature')
plt.title('Weather Forecasting using RNN')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q5: Music Note Generation using RNN. ---\n")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

notes = [60,62,64,65,67,69,71,72]
seq_length = 4
X, y = [], []
for i in range(len(notes)-seq_length):
    X.append(notes[i:i+seq_length])
    y.append(notes[i+seq_length])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(Embedding(128, 32, input_length=seq_length))
model.add(SimpleRNN(64))
model.add(Dense(128, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=200, verbose=1)

seed = [60,62,64,65]
for _ in range(10):
    pred = np.argmax(model.predict(np.array(seed[-seq_length:]).reshape(1,seq_length)), axis=1)[0]
    seed.append(pred)
print("Generated Notes:", seed)
print("____________________________________________________________________________________________________________________________________________")
