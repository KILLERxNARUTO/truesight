from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import librosa
import numpy as np
import sounddevice as sd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter
import random
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

cnn_model = tf.keras.models.load_model(r"model\voice_auth_cnn_model.keras")


app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure secret key

# Initialize MongoDB
client = MongoClient('localhost', 27017)
db = client['voice_authentication']
collection = db['users']

# Audio processing functions
def butter_bandpass(lowcut=100.0, highcut=4000.0, fs=22050, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut=100.0, highcut=4000.0, fs=22050, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def record_audio(duration=5, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return bandpass_filter(audio.flatten(), fs=fs)

def trim_silence(y, sr):
    return librosa.effects.trim(y, top_db=35)[0]

def extract_mfcc(y, sr=22050, max_pad_len=100):
    y = trim_silence(y, sr)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)), mode='constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc[:, :max_pad_len]

def normalize_mfcc(mfcc):
    return (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

# Model functions
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model_from_db():
    users = list(collection.find())
    if not users:
        return None, None

    data, labels = [], []
    label_map = {idx: user['user_id'] for idx, user in enumerate(users)}
    
    for idx, user in enumerate(users):
        for sample in user.get('mfcc_samples', []):
            mfcc = normalize_mfcc(np.array(sample))
            data.append(mfcc)
            labels.append(idx)

    X = np.array(data)[..., np.newaxis]  # Add channel dimension
    y = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_cnn_model((X.shape[1], X.shape[2], 1), y.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))
    return model, label_map

# User management functions
def register_user(user_id, code, num_samples=3):
    mfcc_samples = []
    for i in range(num_samples):
        audio = record_audio()
        if np.max(np.abs(audio)) < 0.01:
            raise ValueError("Recording too quiet")
        mfcc_samples.append(normalize_mfcc(extract_mfcc(audio)).tolist())
    
    collection.update_one(
        {"user_id": user_id},
        {"$set": {"mfcc_samples": mfcc_samples, "code": code}},
        upsert=True
    )

def authenticate_user(user_id, expected_code):
    audio = record_audio()
    if np.max(np.abs(audio)) < 0.01:
        raise ValueError("Recording too quiet")
    
    mfcc = normalize_mfcc(extract_mfcc(audio))
    cnn_input = mfcc.reshape(1, *mfcc.shape, 1)
    
    # Try CNN authentication first
    model, label_map = train_cnn_model_from_db()
    if model:
        pred = model.predict(cnn_input)
        if label_map[np.argmax(pred)] == user_id and np.max(pred) > 0.7:
            return user_id
    
    # Fallback to traditional methods
    user = collection.find_one({"user_id": user_id, "code": expected_code})
    if not user:
        return None
        
    mfcc_flat = mfcc.flatten().reshape(1, -1)
    for sample in user.get('mfcc_samples', []):
        sample_flat = normalize_mfcc(np.array(sample)).flatten().reshape(1, -1)
        
        # Cosine similarity check
        if cosine_similarity(mfcc_flat, sample_flat)[0][0] >= 0.72:
            return user_id
            
        # DTW check
        dist, _ = fastdtw(mfcc.T, np.array(sample).T, dist=euclidean)
        if dist < 300:
            return user_id
    
    return None

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET'])
def login():
    session['verification_code'] = str(random.randint(100000, 999999))
    return render_template('login.html', verification_code=session['verification_code'])

@app.route('/register', methods=['GET'])
def register():
    session['verification_code'] = str(random.randint(100000, 999999))
    return render_template('register.html', verification_code=session['verification_code'])

@app.route('/authenticate', methods=['POST'])
def authenticate():
    user_id = request.form.get('user_id')
    expected_code = session.get('verification_code')
    
    if not expected_code:
        flash("Session expired. Please try again.", "error")
        return redirect(url_for('login'))
    
    try:
        authenticated_id = authenticate_user(user_id, expected_code)
        if authenticated_id:
            flash(f"Welcome {user_id}!", "success")
        else:
            flash("Authentication failed!", "error")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for('login'))

@app.route('/register_voice', methods=['POST'])
def register_voice():
    user_id = request.form.get('user_id')
    expected_code = session.get('verification_code')
    
    if not (user_id and expected_code):
        flash("Missing data", "error")
        return redirect(url_for('register'))
    
    try:
        register_user(user_id, expected_code)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    except Exception as e:
        flash(f"Registration failed: {str(e)}", "error")
        return redirect(url_for('register'))

@app.route('/get_verification_code', methods=['GET'])
def get_verification_code():
    return jsonify({'code': str(random.randint(100000, 999999))})

def get_local_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    local_ip = get_local_ip()
    print(f"\nServer running at:")
    print(f"Local: http://localhost:{port}")
    print(f"Network: http://{local_ip}:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
