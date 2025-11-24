# =========================================================
# --- 0. Import bibliotek ---
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Dropout
)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# --- Konfiguracja lokalnych ścieżek ---
# =========================================================
# Utworzenie lokalnego folderu na wyniki
base_export_dir = './cifar10_exports'
os.makedirs(base_export_dir, exist_ok=True)

print(f"[OK] Folder lokalny utworzony. Pliki będą zapisywane w:\n{os.path.abspath(base_export_dir)}")

# =========================================================
# --- 1. Importowanie danych CIFAR10 ---
# =========================================================
# Wczytanie zbioru danych CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Wyświetlenie przykładowego obrazu (opcjonalne)
print("Etykieta przykładowego obrazu:", y_train[0])
plt.imshow(x_train[0])
plt.axis('off')
plt.title(f'Przykładowy obraz - Klasa: {y_train[0][0]}')
plt.show()

# =========================================================
# --- 2. Wstępne przetwarzanie danych ---
# =========================================================
# Normalizacja do zakresu [0,1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# One-hot encoding etykiet
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# lista nazw klas CIFAR-10 (użyteczna w wyświetleniach)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# =========================================================
# --- 2.5 Data Augmentation ---
# =========================================================
# Augmentacja danych - znacząco poprawia generalizację modelu

datagen = ImageDataGenerator(
    rotation_range=15,           # losowe obroty do 15 stopni
    width_shift_range=0.1,       # przesunięcia w poziomie
    height_shift_range=0.1,      # przesunięcia w pionie
    horizontal_flip=True,        # odbicie lustrzane
    fill_mode='nearest',         # wypełnianie pustych pikseli
    zoom_range=0.1
)

datagen.fit(x_train)

# =========================================================
# --- 3. Definicja ULEPSZONEGO modelu sieci neuronowej ---
# =========================================================

input_shape = (32, 32, 3)  # CIFAR-10 to 32x32 RGB

model = Sequential([
    Input(shape=input_shape),

    # Blok 1 - zredukowany dropout
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Blok 2
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Blok 3
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Klasyfikator - uproszczony
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# =========================================================
# --- 4. Kompilacja modelu ---
# =========================================================
# Optymalizator Adam, funkcja straty categorical_crossentropy, metryka accuracy
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

"""
Prawdziwa klasa: 3 (kot)
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

Predykcja modelu (DOBRA):
y_pred = [0.05, 0.02, 0.08, 0.75, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]
                            ↑ 75% pewności dla kota
"""

# Wyświetlenie podsumowania modelu
model.summary()

# =========================================================
# --- 5. Trenowanie modelu ---
# =========================================================
# Ulepszone callbacki

"""es
Obserwuje val_loss po każdej epoce
Jeśli przez 10 epok loss nie spadnie o minimum 0.0001
Zatrzymuje trening i przywraca najlepsze wagi
"""

"""Po każdej epoce: Model zapisuje wagi do pamięci tymczasowej
Jeśli val_loss jest NAJNIŻSZY: "To są najlepsze wagi do tej pory!"
Po zakończeniu treningu: Przywróć wagi z epoki z najniższym val_loss
"""
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.0001,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

"""rlp
Jak działa:

Jeśli przez 5 epok brak poprawy
Zmniejsz LR o połowę (0.001 → 0.0005 → 0.00025...)
Nie idź poniżej 1e-7
"""
rlp = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_delta=0.0001,
    min_lr=1e-7,
    mode='min',
    verbose=1
)

BATCH_SIZE = 128

print("\n[INFO] Rozpoczynam trening modelu...")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
    epochs=30,
    validation_data=(x_test, y_test_cat),
    callbacks=[es, rlp],
    verbose=1
)

# Timestamp dla nazw plików
ts = datetime.datetime.now().strftime("_%Y%m%d_%H%M")

# =========================================================
# --- 6. Ewaluacja modelu ---
# =========================================================
print("\n[INFO] Ewaluacja modelu...")
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f'Dokładność na zbiorze testowym: {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Strata na zbiorze testowym: {loss:.4f}')

# =========================================================
# --- 7. Wizualizacja przebiegu treningu (loss) ---
# =========================================================
fig_loss_acc = plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss (train)', linewidth=2)
plt.plot(history.history['val_loss'], label='loss (val)', linewidth=2)
plt.xlabel('Epoka', fontsize=12)
plt.ylabel('categorical_crossentropy', fontsize=12)
plt.title(f'Strata na zbiorze testowym: {loss:.4f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy (train)', linewidth=2)
plt.plot(history.history['val_accuracy'], label='accuracy (val)', linewidth=2)
plt.xlabel('Epoka', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f"Dokładność na zbiorze testowym: {accuracy:.4f}", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Zapis wykresu
loss_acc_path = os.path.join(base_export_dir, f'training_loss_accuracy{ts}.png')
fig_loss_acc.savefig(loss_acc_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Zapisano wykres loss/accuracy -> {os.path.abspath(loss_acc_path)}")

plt.show()

# =========================================================
# --- 8. Wizualizacja błędnych klasyfikacji i macierz pomyłek ---
# =========================================================
print("\n[INFO] Generowanie predykcji...")
# Predykcje (etykiety)
pred_probs = model.predict(x_test, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test_cat, axis=1)  # lub po prostu y_test

# Indeksy błędnych klasyfikacji
incorrect_indices = np.nonzero(pred_labels != true_labels)[0]
print(f"[INFO] Liczba błędnych klasyfikacji: {len(incorrect_indices)} / {len(y_test)} ({len(incorrect_indices)/len(y_test)*100:.2f}%)")

# Wyświetlenie kilku błędnych przykładów
n_show = min(6, len(incorrect_indices))
if n_show > 0:
    fig_errors, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_show):
        idx = incorrect_indices[i]
        axes[i].imshow(x_test[idx])
        axes[i].set_title(f"Prawda: {class_names[true_labels[idx]]}\nPredykcja: {class_names[pred_labels[idx]]}", 
                         fontsize=10, color='red')
        axes[i].axis('off')
    
    # Ukryj puste subploty jeśli jest mniej niż 6 błędów
    for i in range(n_show, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Zapis błędnych przykładów
    errors_path = os.path.join(base_export_dir, f'misclassified_examples{ts}.png')
    fig_errors.savefig(errors_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Zapisano błędne przykłady -> {os.path.abspath(errors_path)}")
    plt.show()

# Confusion matrix
print("\n[INFO] Generowanie macierzy pomyłek...")
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
plt.title('Macierz pomyłek - CIFAR-10', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Zapis macierzy pomyłek
cm_path = os.path.join(base_export_dir, f'confusion_matrix{ts}.png')
fig_cm.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"[OK] Zapisano macierz pomyłek -> {os.path.abspath(cm_path)}")
plt.show()

# =========================================================
# --- 9. Zapis modelu ---
# =========================================================
model_path = os.path.join(base_export_dir, f'my_model{ts}.keras')
model.save(model_path)
print(f"\n[OK] Model zapisany -> {os.path.abspath(model_path)}")

# =========================================================
# --- 10. Podsumowanie ---
# =========================================================
print("\n" + "="*60)
print("PODSUMOWANIE TRENINGU")
print("="*60)
print(f"Dokładność (accuracy): {accuracy*100:.2f}%")
print(f"Strata (loss): {loss:.4f}")
print(f"Liczba epok: {len(history.history['loss'])}")
print(f"Błędne klasyfikacje: {len(incorrect_indices)} / {len(y_test)}")
print(f"\nWszystkie pliki zapisane w: {os.path.abspath(base_export_dir)}")
print("="*60)