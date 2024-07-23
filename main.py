import cv2
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def load_images(folder, target_size=(150, 150)):
    images = []
    labels = []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}

    for label in label_map.keys():
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            image_path = os.path.join(path, filename)
            image = None

            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image = cv2.imread(image_path)
            elif filename.lower().endswith('dcm'):
                dcm = pydicom.dcmread(image_path)
                image = dcm.pixel_array
            if image is not None:
                image = cv2.resize(image, target_size)
                image = image / 255.0
                images.append(image)
                labels.append(label_map[label])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def shuffle_data(images, labels, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=test_ratio + val_ratio, random_state=0, stratify=labels)

    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=test_ratio / (test_ratio + val_ratio), random_state=0, stratify=temp_labels)

    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def train_model():
    global model, history
    images, labels = load_images(total_dir)
    train_image, train_label, test_image, test_label, val_image, val_label = shuffle_data(images, labels)

    train_label = to_categorical(train_label, num_classes=2)
    val_label = to_categorical(val_label, num_classes=2)
    test_label = to_categorical(test_label, num_classes=2)

    print(f"Train image: {train_image.shape}")
    print(f"Train label: {train_label.shape}")
    print(f"Validation image: {val_image.shape}")
    print(f"Validation label: {val_label.shape}")
    print(f"Test image: {test_image.shape}")
    print(f"Test label: {test_label.shape}")

    early_stopping = EarlyStopping(
        min_delta = 0.00001,
        patience = 5,
        restore_best_weights = True
    )

    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, kernel_size=3, activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, kernel_size=3, activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_image,
        train_label,
        epochs=100,
        batch_size=256,
        validation_data=(val_image, val_label),
        verbose=1,
        callbacks=[early_stopping]
    )

def test_model():
    global model
    test_loss, test_accuracy = model.evaluate(test_image, test_label)
    print(f"Test accuracy: {test_accuracy}")
    predictions = model.predict(test_image)
    global predicted_labels
    predicted_labels = np.argmax(predictions, axis=1)

def display_training_history(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_num = range(1, len(accuracy) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_num, accuracy, 'r', label='Training accuracy')
    plt.plot(epoch_num, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.subplot(2, 1, 2)
    plt.plot(epoch_num, loss, 'r', label='Training loss')
    plt.plot(epoch_num, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    class_names = ['NORMAL', 'PNEUMONIA']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def menu():
    print("Menu:")
    print("1. Train Model")
    print("2. Save Model")
    print("3. Load Model")
    print("4. Display Training History")
    print("5. Display Confusion Matrix")
    print("6. Exit")

def display_history():
    global history
    display_training_history(history)

def display_confusion():
    global predicted_labels, test_label
    plot_confusion_matrix(np.argmax(test_label, axis=1), predicted_labels)

def main():
    while True:
        menu()
        choice = input("Enter:")
        if choice == '1':
            train_model()
        elif choice == '2':
            model.save('demo_model.keras')
            print("Model saved.")
        elif choice == '3':
            model = load_model('demo_model.keras')
            print("Model loaded.")
        elif choice == '4':
            display_history()
        elif choice == '5':
            display_confusion()
        elif choice == '6':
            break
        else:
            print("Select between 1-6")

if __name__ == "__main__":
    total_dir = '../input/total-dataset/Total Dataset'
    model = None
    history = None
    test_label = None
    predicted_labels = None
    main()
