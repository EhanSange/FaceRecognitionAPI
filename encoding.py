import face_recognition
import cv2
import os
import pickle

path = 'Model_train'
images = []
classNames = []

# Fungsi untuk membaca semua gambar dari subdirektori
def load_images_from_subdirs(path):
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                curImg = cv2.imread(img_path)
                if curImg is not None:
                    images.append(curImg)
                    classNames.append(subdir)

# Panggil fungsi untuk membaca gambar
load_images_from_subdirs(path)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encode = encodes[0]
            encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

# Simpan encoding dan nama ke file
with open('encodings.pickle', 'wb') as f:
    pickle.dump((encodeListKnown, classNames), f)
