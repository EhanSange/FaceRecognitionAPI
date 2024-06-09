import face_recognition
import cv2
import numpy as np
import pickle

# Muat encoding wajah yang telah dilatih dan nama kelas
with open('encodings.pickle', 'rb') as f:
    encodeListKnown, classNames = pickle.load(f)

# Ambang batas untuk menentukan apakah wajah dikenali atau tidak
tolerance = 0.6

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Temukan semua wajah dalam frame dan encoding mereka
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Bandingkan wajah yang terdeteksi dengan yang ada di data latih
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "TIDAK DIKENALI"
        
        # Gambar kotak di sekitar wajah dan tulis nama
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
