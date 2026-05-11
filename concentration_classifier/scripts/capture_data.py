import cv2
import os
import time

def create_dataset():
    user_name = input("Entrez le nom de la personne à capturer : ").strip().lower()
    if not user_name:
        print("Erreur : Le nom est obligatoire.")
        return

    base_path = f"data/raw/{user_name}"
    classes = ["concentre", "distrait", "absent"]
    
    for cls in classes:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"--- Capture pour : {user_name.upper()} ---")
    print("Appuyez sur : 'c' (Concentre), 'd' (Distrait), 'a' (Absent), 'q' (Quitter)")

    counts = {cls: len(os.listdir(os.path.join(base_path, cls))) for cls in classes}

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.putText(frame, f"User: {user_name} | C:{counts['concentre']} D:{counts['distrait']} A:{counts['absent']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Capture Multi-User", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): save_img(frame, base_path, "concentre", counts)
        elif key == ord('d'): save_img(frame, base_path, "distrait", counts)
        elif key == ord('a'): save_img(frame, base_path, "absent", counts)

    cap.release()
    cv2.destroyAllWindows()

def save_img(frame, base_path, label, counts):
    filename = f"{label}_{int(time.time() * 1000)}.jpg"
    filepath = os.path.join(base_path, label, filename)
    cv2.imwrite(filepath, frame)
    counts[label] += 1
    print(f"Enregistré dans {label}")

if __name__ == "__main__":
    create_dataset()
