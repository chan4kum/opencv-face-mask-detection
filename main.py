import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mask_detector.keras")
face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in face_c.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
            face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))[None].astype("float32")
            p = float(model.predict(face, verbose=0)[0][0])  # probability of without_mask
            label, color = ("No Mask", (0, 0, 255)) if p > 0.5 else ("Mask", (0, 255, 0))
            conf = p if p > 0.5 else 1 - p
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
