from flask import Flask, request, jsonify
from pymongo import MongoClient
from geopy.distance import geodesic
from dotenv import load_dotenv
import cv2
import face_recognition
import numpy as np
import os
import pickle


app = Flask(__name__)

load_dotenv()
db = MongoClient(os.getenv('MONGODB_URL'))
collection = db['location']

@app.route('/location', methods=['GET'])
def get_location():
    location = collection.find_one()
    
    return jsonify({
        'latitude': location.get('latitude'),
        'longitude': location.get('longitude'),
        'radius': location.get('radius')
    })

@app.route('/location', methods=['PUT'])
def update_location():    
    data = request.get_json()
    secret = data.get('secret')

    if secret != os.getenv('SECRET_KEY'):
        return jsonify({'status': 'Failed', 'message': 'Unauthorized'}), 401
    
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    radius = data.get('radius')
    
    collection.update_one({}, {'$set': {
        'latitude': latitude,
        'longitude': longitude,
        'radius': radius
    }}, upsert=True)
    
    return jsonify({'status': 'Success', 'message': 'Detil lokasi berhasil diperbaharui'})

@app.route('/location', methods=['POST'])
def check_location():
    data = request.get_json()
    user_location = (data.get('latitude'), data.get('longitude'))

    location = collection.find_one()
    allowed_location = (location.get('latitude'), location.get('longitude'))
    allowed_radius = location.get('radius')
    distance = geodesic(allowed_location, user_location).meters

    if distance <= allowed_radius:
        return jsonify({'status': 'Success', 'message': 'Anda berada di sekolah'})
    else:
        return jsonify({'status': 'Failed', 'message': 'Anda berada di luar jangkauan'})

@app.route('/face-matching', methods=['POST'])
def face_matching():
    image_data = request.files['image']

    if request.files['image'].filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    with open('encodings.pickle', 'rb') as f:
        encodeListKnown, classNames = pickle.load(f)

    img = face_recognition.load_image_file(image_data)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    detected_name = ''

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
        encode = face_recognition.face_encodings(face_rgb)
        if len(encode) == 0:
            continue
        encode = encode[0]

        distances = np.linalg.norm(encodeListKnown - encode, axis=1)
        matchIndex = np.argmin(distances)
        min_distance = distances[matchIndex]

        if min_distance < 0.8:
            detected_name = classNames[matchIndex].upper()

    if detected_name != '':
        return jsonify({'status': 'Success', 'message': detected_name})
    else:
        return jsonify({'status': 'Failed', 'message':'Wajah tidak dikenali'})

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=8000)