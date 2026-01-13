print("\n--- Q1: Deep Neural Network (Pass/Fail Prediction) ---\n")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
n = 2000
study_hours = np.clip(np.random.normal(4, 2, n), 0, 10)
attendance = np.clip(np.random.normal(80, 10, n), 40, 100)
score = 0.6 * study_hours + 0.4 * (attendance / 10.0)
y = (score > 7.5).astype(int)
X = np.column_stack([study_hours, attendance])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_deep_model(neurons=(64, 32, 16)):
    m = Sequential([Input(shape=(2,))])
    for u in neurons:
        m.add(Dense(u, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_shallow_model(units=32):
    m = Sequential([Input(shape=(2,)), Dense(units, activation="relu"), Dense(1, activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q2: Activation Function Analysis ---\n")
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data.astype("float32")
y = to_categorical(iris.target, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=iris.target)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(act):
    m = Sequential([Input(shape=(4,)), Dense(64, activation=act), Dense(32, activation=act), Dense(16, activation=act), Dense(3, activation="softmax")])
    m.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return m

m_relu = build_model("relu")
m_relu.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
acc_relu = m_relu.evaluate(X_test, y_test, verbose=0)[1]

m_tanh = build_model("tanh")
m_tanh.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
acc_tanh = m_tanh.evaluate(X_test, y_test, verbose=0)[1]

print("ReLU:", acc_relu, "tanh:", acc_tanh)

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q3: Hyperparameter Tuning in DNN ---\n")
from tensorflow.keras.datasets import mnist
import time
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,784)/255.0
x_test=x_test.reshape(-1,784)/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def build(hidden,neurons):
    m=Sequential([Input(shape=(784,))])
    for _ in range(hidden):
        m.add(Dense(neurons,activation="relu"))
    m.add(Dense(10,activation="softmax"))
    m.compile(optimizer=Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    return m

configs=[{"hidden":2,"neurons":64,"batch":32},{"hidden":4,"neurons":128,"batch":64},{"hidden":6,"neurons":256,"batch":128}]
for c in configs:
    m=build(c["hidden"],c["neurons"])
    t0=time.time()
    m.fit(x_train,y_train,epochs=5,batch_size=c["batch"],verbose=0,validation_split=0.1)
    dur=time.time()-t0
    acc=m.evaluate(x_test,y_test,verbose=0)[1]
    print(c,"time",round(dur,2),"acc",round(acc,4))

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q4: Overfitting and Regularization ---\n")
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train.reshape(-1,784)/255.0
x_test=x_test.reshape(-1,784)/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def overfit_model():
    m=Sequential([Input(shape=(784,)),Dense(512,activation="relu"),Dense(512,activation="relu"),Dense(512,activation="relu"),Dense(10,activation="softmax")])
    m.compile(optimizer=Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    return m

def reg_model():
    m=Sequential([Input(shape=(784,)),Dense(512,activation="relu"),Dropout(0.5),Dense(256,activation="relu"),Dropout(0.5),Dense(128,activation="relu"),Dense(10,activation="softmax")])
    m.compile(optimizer=Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    return m

m1=overfit_model()
m1.fit(x_train,y_train,epochs=20,batch_size=64,verbose=0,validation_split=0.2)
acc1=m1.evaluate(x_test,y_test,verbose=0)[1]

m2=reg_model()
es=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
m2.fit(x_train,y_train,epochs=30,batch_size=64,verbose=0,validation_split=0.2,callbacks=[es])
acc2=m2.evaluate(x_test,y_test,verbose=0)[1]

print("Overfit acc:",acc1,"Regularized acc:",acc2)
