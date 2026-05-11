import cv2
import os
import time

def create_dataset():
    base_path = "data/raw"
    classes = ["concentre", "distrait", "absent"]
    
    for cls in classes:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return

    print("--- Mode Capture de Dataset ---")
    print("Appuyez sur :")
    print("  'c' pour CONCENTRÉ")
    print("  'd' pour DISTRAIT")
    print("  'a' pour ABSENT")
    print("  'q' pour QUITTER")

    counts = {cls: len(os.listdir(os.path.join(base_path, cls))) for cls in classes}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"C: {counts['concentre']} | D: {counts['distrait']} | A: {counts['absent']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Capture - Dataset Concentration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            save_img(frame, base_path, "concentre", counts)
        elif key == ord('d'):
            save_img(frame, base_path, "distrait", counts)
        elif key == ord('a'):
            save_img(frame, base_path, "absent", counts)

    cap.release()
    cv2.destroyAllWindows()

def save_img(frame, base_path, label, counts):
    filename = f"{label}_{int(time.time() * 1000)}.jpg"
    filepath = os.path.join(base_path, label, filename)
    cv2.imwrite(filepath, frame)
    counts[label] += 1
    print(f"Image enregistrée : {filepath}")

if __name__ == "__main__":
    create_dataset()
