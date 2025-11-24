# 🖼️ CIFAR-10 CNN Image Classification

Profesjonalny projekt klasyfikacji obrazów z wykorzystaniem głębokiej sieci konwolucyjnej (CNN) trenowanej na zbiorze danych CIFAR-10. Implementacja w **TensorFlow/Keras** z pełną augmentacją danych i zaawansowanymi callbackami treningowymi.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Wymagania](#-wymagania)
- [Instalacja](#-instalacja)
- [Uruchomienie](#-uruchomienie)
- [Struktura projektu](#-struktura-projektu)
- [Architektura modelu](#-architektura-modelu)
- [Wyniki](#-wyniki)
- [Wizualizacje](#-wizualizacje)
- [Autor](#-autor)

---

## 🎯 Opis projektu

Projekt implementuje zaawansowaną sieć neuronową CNN do klasyfikacji obrazów z zestawu **CIFAR-10**. Kod został zaprojektowany do uruchomienia **lokalnie** (Visual Studio, PyCharm, VSCode) bez konieczności korzystania z Google Colab czy chmurowych dysków.

### Dataset: CIFAR-10
- **60 000 kolorowych obrazów** (32x32 piksele, RGB)
- **10 klas**: samolot, samochód, ptak, kot, jeleń, pies, żaba, koń, statek, ciężarówka
- Automatyczne pobieranie przez Keras

---

## ✨ Funkcjonalności

- ✅ **Data Augmentation** - rotacje, przesunięcia, odbicia lustrzane, zoom
- ✅ **Batch Normalization** - stabilizacja treningu
- ✅ **Dropout** - zapobieganie przeuczeniu
- ✅ **Early Stopping** - automatyczne zatrzymanie przy braku poprawy
- ✅ **Learning Rate Reduction** - dynamiczne dostosowanie tempa uczenia
- ✅ **Wizualizacje**:
  - Wykresy loss/accuracy
  - Macierz pomyłek (confusion matrix)
  - Przykłady błędnych klasyfikacji
- ✅ **Automatyczny zapis** modelu i wszystkich wykresów

---

## 🔧 Wymagania

### Oprogramowanie
- **Python** >= 3.7
- **pip** (menedżer pakietów)

### Biblioteki Python
```bash
tensorflow >= 2.0
numpy
matplotlib
scikit-learn
```

---

## 📦 Instalacja

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/KieltRadek/Image_Recognition_CIFAR10.git
cd Image_Recognition_CIFAR10
```

### 2. (Opcjonalnie) Utwórz wirtualne środowisko
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Zainstaluj zależności
```bash
pip install tensorflow scikit-learn matplotlib numpy
```

**LUB** (jeśli masz plik `requirements.txt`):
```bash
pip install -r requirements.txt
```

---

## 🚀 Uruchomienie

```bash
python cifar10_local.py
```

### Co się dzieje podczas uruchomienia?
1. ⬇️ Automatyczne pobieranie datasetu CIFAR-10 (jednorazowo)
2. 🔄 Przetwarzanie i normalizacja danych
3. 🧠 Budowa architektury CNN
4. 🏋️ Trening modelu (domyślnie do 30 epok z early stopping)
5. 📊 Generowanie wykresów i statystyk
6. 💾 Zapis modelu i wizualizacji w `./cifar10_exports/`

---

## 📁 Struktura projektu

```
Image_Recognition_CIFAR10/
│
├── cifar10_local.py              # Główny skrypt treningowy
├── README.md                     # Dokumentacja projektu
├── requirements.txt              # Lista zależności (opcjonalnie)
│
└── cifar10_exports/              # 📂 Folder z wynikami (tworzony automatycznie)
    ├── my_model_YYYYMMDD_HHMM.keras
    ├── training_loss_accuracy_YYYYMMDD_HHMM.png
    ├── confusion_matrix_YYYYMMDD_HHMM.png
    └── misclassified_examples_YYYYMMDD_HHMM.png
```

---

## 🧠 Architektura modelu

### CNN - 3 bloki konwolucyjne

```
Input (32x32x3)
    ↓
┌─────────────────────┐
│  BLOK 1             │
│  - Conv2D (32)      │
│  - BatchNorm        │
│  - ReLU             │
│  - Conv2D (32)      │
│  - BatchNorm        │
│  - ReLU             │
│  - MaxPooling2D     │
│  - Dropout (0.2)    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  BLOK 2             │
│  - Conv2D (64)      │
│  - BatchNorm        │
│  - ReLU             │
│  - Conv2D (64)      │
│  - BatchNorm        │
│  - ReLU             │
│  - MaxPooling2D     │
│  - Dropout (0.25)   │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  BLOK 3             │
│  - Conv2D (128)     │
│  - BatchNorm        │
│  - ReLU             │
│  - Conv2D (128)     │
│  - BatchNorm        │
│  - ReLU             │
│  - MaxPooling2D     │
│  - Dropout (0.3)    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  KLASYFIKATOR       │
│  - Flatten          │
│  - Dense (128)      │
│  - BatchNorm        │
│  - ReLU             │
│  - Dropout (0.4)    │
│  - Dense (10)       │
│  - Softmax          │
└─────────────────────┘
    ↓
Output (10 klas)
```

**Parametry treningu:**
- Optymalizator: **Adam** (learning rate: 0.001)
- Funkcja straty: **Categorical Crossentropy**
- Batch size: **128**
- Epoki: do **30** (z early stopping)

---

## 📊 Wyniki

Typowe wyniki po treningu (może się różnić w zależności od inicjalizacji):

| Metryka | Wartość |
|---------|---------|
| **Accuracy (test)** | ~75-82% |
| **Loss (test)** | ~0.55-0.75 |
| **Liczba parametrów** | ~500K |

---

## 📈 Wizualizacje

Po zakończeniu treningu w folderze `cifar10_exports/` znajdziesz:

### 1. **Wykresy Loss & Accuracy**
Wizualizacja procesu treningu pokazująca przebieg funkcji straty i dokładności na zbiorach treningowym i walidacyjnym.

### 2. **Macierz pomyłek**
Szczegółowa analiza błędów klasyfikacji - pokazuje które klasy są najczęściej mylone ze sobą.

### 3. **Błędne klasyfikacje**
Przykłady obrazów, które model sklasyfikował niepoprawnie, z prawdziwymi i przewidywanymi etykietami.

---

## 🛠️ Konfiguracja (opcjonalna)

Możesz dostosować parametry w pliku `cifar10_local.py`:

```python
# Liczba epok
epochs=30  # Zmień na większą/mniejszą wartość

# Batch size
BATCH_SIZE = 128  # Zmniejsz jeśli masz mało RAM (np. 64)

# Learning rate
optimizer=Adam(learning_rate=1e-3)  # Dostosuj tempo uczenia

# Data Augmentation
rotation_range=15  # Zakres rotacji obrazów
```

---

## 📝 Przykład użycia wytrenowanego modelu

```python
from tensorflow import keras
import numpy as np

# Wczytaj model
model = keras.models.load_model('./cifar10_exports/my_model_YYYYMMDD_HHMM.keras')

# Przygotuj obraz (32x32x3, znormalizowany)
img = ...  # Twój obraz
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# Predykcja
prediction = model.predict(img)
class_idx = np.argmax(prediction)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
print(f"Predykcja: {class_names[class_idx]}")
```

---

## 🤝 Współpraca

Chętnie przyjmę pull requesty! Jeśli chcesz dodać nowe funkcje:

1. Zforkuj projekt
2. Stwórz branch (`git checkout -b feature/NowaFunkcja`)
3. Commit (`git commit -m 'Dodano NowaFunkcja'`)
4. Push (`git push origin feature/NowaFunkcja`)
5. Otwórz Pull Request

---

## 📄 Licencja

Projekt udostępniony na licencji **MIT**.

---

## 👨‍💻 Autor

**KieltRadek**

- GitHub: [@KieltRadek](https://github.com/KieltRadek)

---

## ⭐ Podziękowania

Jeśli projekt Ci się podoba - zostaw gwiazdkę! ⭐

---

**Made with ❤️ and TensorFlow**