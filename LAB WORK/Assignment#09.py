print("\n--- Q1: Build a CNN to classify images of Cats and Dogs.---\n")
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "DataFiles\dataset"  
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------
# Data generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print("Class indices:", train_gen.class_indices)

# -------------------------------
# Build CNN model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# Train model
# -------------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# -------------------------------
# Plot accuracy and loss
# -------------------------------
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png', dpi=150)
    plt.show()

plot_history(history)

# -------------------------------
# Confusion matrix
# -------------------------------
val_gen.reset()
pred_probs = model.predict(val_gen)
preds = (pred_probs > 0.5).astype(int).ravel()
true_labels = val_gen.classes

cm = confusion_matrix(true_labels, preds)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_labels, preds, target_names=list(val_gen.class_indices.keys())))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(val_gen.class_indices.keys()),
            yticklabels=list(val_gen.class_indices.keys()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# -------------------------------
# Predict a test image
# -------------------------------
def predict_image(model, img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prob = model.predict(x)[0][0]
    label = "DOG" if prob >= 0.5 else "CAT"
    return label, prob

# Pick a test image from validation set
test_image_path = val_gen.filepaths[0]
pred_label, pred_prob = predict_image(model, test_image_path)

print(f"Test image: {test_image_path}")
print(f"Predicted: {pred_label} (prob={pred_prob:.4f})")

# Final verdict
if pred_label == "CAT":
    print("This image is a CAT")
else:
    print("This image is a DOG")
