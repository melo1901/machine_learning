import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets, layers, models
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

# Zaladowanie danych
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizacja danych
train_images, test_images = train_images / 255.0, test_images / 255.0

# Podzial danych na zbiory treningowe i walidacyjne
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Zdefiniowanie modelu  
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Wyrownanie obrazow wejsciowych
    layers.Dense(128, activation='relu'),   # Warstwa ukryta z 128 neuronami
    layers.Dropout(0.2),                    # Warstwa Dropout
    layers.Dense(10, activation='softmax')  # Warstwa wyjsciowa z 10 neuronami (10 klas dla 10 cyfr)
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Podsumowanie modelu
model.summary()


# Trenowanie modelu 
history = model.fit(
    train_images, train_labels,
    epochs=10,  # Liczba epok
    validation_data=(val_images, val_labels)
)

# Ocena modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')



# Wizualizacja wynikow trenowania
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Wizualizacja wynikow trenowania   
plot_training_history(history)


# Predykcja na zbiorze testowym
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Raport klasyfikacji
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))

# Krzywa ROC
def plot_roc_curve(test_labels, predictions):
    plt.figure(figsize=(8, 8))
    for i in range(10):  # 10 klas dla 10 cyfr
        fpr, tpr, _ = roc_curve((test_labels == i).astype(int), predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each Class')
    plt.legend()
    plt.show()

# Naszkicowanie krzywej ROC
plot_roc_curve(test_labels, predictions)




# Zdefiniowanie zoptymalizowanego modelu (z warstwami konwolucyjnymi)
optimized_model = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Kompilacja zoptymalizowanego modelu
optimized_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# Trening zoptymalizowanego modelu
optimized_history = optimized_model.fit(
    train_images[..., np.newaxis], train_labels,
    epochs=10,
    validation_data=(val_images[..., np.newaxis], val_labels)
)

# Ocena zoptymalizowanego modelu na zbiorze testowym
test_loss, test_acc = optimized_model.evaluate(test_images[..., np.newaxis], test_labels)
print(f'Test accuracy (optimized model): {test_acc}')

# Wizualizacja wynikow trenowania zoptymalizowanego modelu
plot_training_history(optimized_history)