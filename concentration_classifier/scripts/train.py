import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path="data/raw", size=(64, 64)):
    images = []
    labels = []
    classes = ["concentre", "distrait", "absent"]
    class_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        path = os.path.join(data_path, cls)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, size)
                images.append(resized.flatten())
                labels.append(class_map[cls])
            except Exception as e:
                print(f"Erreur sur l'image {img_name}: {e}")

    return np.array(images) / 255.0, np.array(labels)

def train_and_evaluate():
    print("Chargement des données...")
    X, y = load_data()
    
    if len(X) == 0:
        print("Erreur : Aucune donnée trouvée. Lancez d'abord capture_data.py")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "k-NN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"\nEntraînement de {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = model
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred, target_names=["concentre", "distrait", "absent"]))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=["C", "D", "A"], yticklabels=["C", "D", "A"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        plt.savefig(f"models/cm_{name.replace(' ', '_').lower()}.png")
        plt.close()

    joblib.dump(models["Random Forest"], "models/best_model.pkl")
    print("\nMeilleur modèle sauvegardé dans models/best_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
