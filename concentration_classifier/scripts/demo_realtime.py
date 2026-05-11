import cv2
import numpy as np
import joblib
import os
from collections import deque

def run_demo():
    try:
        name_model = joblib.load("models/name_model.pkl")
        status_model = joblib.load("models/status_model.pkl")
        user_list = joblib.load("models/user_list.pkl")
    except:
        print("Erreur : Modèles manquants.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    status_classes = ["CONCENTRE", "DISTRAIT", "ABSENT"]
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] 

    # Lissage (Positions + Prédictions)
    smooth_box = deque(maxlen=8)
    smooth_id = deque(maxlen=10)
    smooth_st = deque(maxlen=10)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. IA : Prédiction
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.flatten().reshape(1, -1) / 255.0
        
        id_raw = name_model.predict(normalized)[0]
        st_raw = status_model.predict(normalized)[0]
        
        smooth_id.append(id_raw)
        smooth_st.append(st_raw)
        
        final_id = max(set(smooth_id), key=list(smooth_id).count)
        final_st = max(set(smooth_st), key=list(smooth_st).count)
        
        name = user_list[final_id].upper()
        label = status_classes[final_st]
        color = colors[final_st]

        # 2. Détection Visage fluide
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            smooth_box.append(faces[0])
            avg_box = np.mean(smooth_box, axis=0).astype(int)
            x, y, w, h = avg_box
            
            # Carré fin + Nom sur le carré
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            cv2.putText(frame, name, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 3. Texte d'état fixe en haut à gauche
        cv2.putText(frame, f"ETAT: {label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("IA Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
