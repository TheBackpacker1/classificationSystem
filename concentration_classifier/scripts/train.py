import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_multi_user_data(data_root="data/raw", size=(64, 64)):
    images = []
    name_labels = []
    status_labels = []
    
    users = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    user_map = {name: i for i, name in enumerate(users)}
    status_map = {"concentre": 0, "distrait": 1, "absent": 2}
    
    print(f"Utilisateurs trouvés : {users}")

    for user in users:
        user_path = os.path.join(data_root, user)
        for status in status_map.keys():
            status_path = os.path.join(user_path, status)
            if not os.path.exists(status_path): continue
            
            for img_name in os.listdir(status_path):
                try:
                    img = cv2.imread(os.path.join(status_path, img_name))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, size)
                    
                    images.append(resized.flatten())
                    name_labels.append(user_map[user])
                    status_labels.append(status_map[status])
                except: continue

    return np.array(images)/255.0, np.array(name_labels), np.array(status_labels), users

def train():
    X, y_name, y_status, user_list = load_multi_user_data()
    if len(X) == 0:
        print("Aucune donnée !")
        return

    print("Entraînement du modèle d'identité...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_name, test_size=0.2, random_state=42)
    name_model = RandomForestClassifier(n_estimators=100)
    name_model.fit(X_train, y_train)
    print(f"Accuracy Identité : {accuracy_score(y_test, name_model.predict(X_test)):.2f}")

    print("Entraînement du modèle d'état...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_status, test_size=0.2, random_state=42)
    status_model = RandomForestClassifier(n_estimators=100)
    status_model.fit(X_train, y_train)
    print(f"Accuracy État : {accuracy_score(y_test, status_model.predict(X_test)):.2f}")

    joblib.dump(name_model, "models/name_model.pkl")
    joblib.dump(status_model, "models/status_model.pkl")
    joblib.dump(user_list, "models/user_list.pkl")
    print("\nModèles sauvegardés !")

if __name__ == "__main__":
    train()
