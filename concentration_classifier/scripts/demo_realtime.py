import cv2
import numpy as np
import joblib
import os

def run_demo():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle {model_path} n'existe pas. Lancez d'abord train.py")
        return

    model = joblib.load(model_path)
    classes = ["CONCENTRE", "DISTRAIT", "ABSENT"]
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] 

    cap = cv2.VideoCapture(0)
    
    print("--- Démo Temps Réel ---")
    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.flatten().reshape(1, -1) / 255.0

        prediction = model.predict(normalized)[0]
        label = classes[prediction]
        color = colors[prediction]

        cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"ETAT: {label}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Detection de Concentration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
