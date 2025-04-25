import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ----------- Configuration -----------

DATASET_DIR = "afhq/train"  # Change this to your dataset path
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 100

# ----------- Load and Preprocess Data -----------

def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            continue
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(CATEGORIES.index(category))
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                continue
    return np.array(data), np.array(labels)

print("Loading data...")
X, y = load_data()
X = X / 255.0
y_cat = to_categorical(y, num_classes=2)

# Train-test split for CNN
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ----------- Define and Train CNN Model -----------

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

print("Training CNN...")
cnn_history = cnn_model.fit(X_train, y_train, epochs=10, validation_split=0.2)
print("CNN training completed.")

# Evaluate CNN
print("Evaluating CNN...")
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cnn_report = classification_report(y_true, y_pred_cnn, target_names=CATEGORIES)
print("CNN Classification Report:")
print(cnn_report)

cnn_cm = confusion_matrix(y_true, y_pred_cnn)

# ----------- Traditional ML Models -----------

# Flatten image data
X_flat = X.reshape(len(X), -1)
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(X_flat, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_flat, y_train_flat)
    y_pred = model.predict(X_test_flat)
    acc = np.mean(y_pred == y_test_flat)
    cm = confusion_matrix(y_test_flat, y_pred)
    results[name] = {"model": model, "confusion_matrix": cm, "accuracy": acc}
    print(f"{name} Accuracy: {acc:.4f}")

# Add CNN to results
cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
results["CNN"] = {"accuracy": cnn_accuracy, "confusion_matrix": cnn_cm}

# ----------- Plot Confusion Matrices -----------

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=axs[i])
    axs[i].set_title(f"{name} - Confusion Matrix")
    axs[i].set_xlabel("Predicted")
    axs[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png")
plt.show()

# ----------- Bar Plot: Accuracy Comparison -----------

plt.figure(figsize=(8, 6))
model_names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in model_names]
sns.barplot(x=model_names, y=accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()
