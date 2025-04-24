import os
import warnings
import time
from collections import Counter
import random
import tempfile
import numpy as np
import librosa
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, render_template, redirect
from flask_cors import CORS
from pymongo import MongoClient
from pydub import AudioSegment
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import butter, lfilter
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# -------------------------
# Environment + Paths
# -------------------------
load_dotenv()

ffmpeg_path = os.getenv("FFMPEG_PATH")
ffprobe_path = os.getenv("FFPROBE_PATH")

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

if not ffmpeg_path or not os.path.exists(ffmpeg_path):
    raise FileNotFoundError(f"FFmpeg path not found: {ffmpeg_path}")
if not ffprobe_path or not os.path.exists(ffprobe_path):
    raise FileNotFoundError(f"FFprobe path not found: {ffprobe_path}")

default_model_path = os.getenv("MODEL_PATH")
default_cnn_model = load_model(default_model_path)

# -------------------------
# Flask + MongoDB Setup
# -------------------------
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=[
    "http://localhost:5000",
    "https://d7e9-2409-40f4-3e-a641-9569-a154-d595-e5a0.ngrok-free.app"
])
app.secret_key = os.getenv("SECRET_KEY", "super_secret")

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client["voice_authentication"]
users_collection = db["users"]
users_collection.create_index("user_id", unique=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------
# Audio + CNN Utils
# -------------------------
def convert_webm_to_wav(audio_file):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_input:
        audio_file.save(temp_input.name)
        webm_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        wav_path = temp_output.name

    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {e}")
    finally:
        os.remove(webm_path)

    return wav_path

def trim_silence(y, sr):
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

def butter_bandpass(lowcut=100.0, highcut=4000.0, fs=22050, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut=100.0, highcut=4000.0, fs=22050, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_mfcc(y, sr=22050, max_pad_len=100):
    y = trim_silence(y, sr)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # always 40
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T  # shape: (100, 40)



def normalize_mfcc(mfcc):
    return (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model_from_db():
    data = []
    labels = []
    label_map = {}
    users = list(users_collection.find())

    for idx, user in enumerate(users):
        user_id = user['user_id']
        label_map[idx] = user_id
        mfcc_samples = user.get('mfcc_samples', [])
        for sample in mfcc_samples:
            mfcc_array = np.array(sample)
            if mfcc_array.shape != (100, 40):
                mfcc_array = np.resize(mfcc_array, (100, 40))
            mfcc_array = normalize_mfcc(mfcc_array)
            data.append(mfcc_array)
            labels.append(idx)

    if len(set(labels)) < 2:
        print("❌ Not enough distinct users to train CNN. Using fallback model.")
        return default_cnn_model, {0: 'default'}


    X = np.array(data)[..., np.newaxis]
    y = to_categorical(labels)

    print("📊 Label distribution:", dict(Counter(labels)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = create_cnn_model(X.shape[1:], y.shape[1])

    print("\n📊 CNN Training Started...\n")
    history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_test, y_test), verbose=1)

    print("\n✅ Training Complete")
    print("🧠 Final Accuracy:", history.history['accuracy'][-1])
    print("🧪 Final Val Accuracy:", history.history['val_accuracy'][-1])

    return model, label_map
    X = np.array(data)[...,np.newaxis]
    y = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = create_cnn_model(X.shape[1:], y.shape[1])
    print("\n📊 CNN Training Started...\n")
    history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_test, y_test), verbose=1)
    print("\n✅ Training Complete")
    print("🧠 Final Accuracy:", history.history['accuracy'][-1])
    print("🧪 Final Val Accuracy:", history.history['val_accuracy'][-1])
    return model, label_map

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return redirect("/login")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/api/verification-code", methods=["GET"])
def get_code():
    user_id = request.args.get("user_id")
    code = str(random.randint(100000, 999999))
    session["verification_code"] = code
    session["code_created_at"] = int(time.time())
    session["code_attempts"] = 0
    print(f"Verification code for {user_id}: {code}")
    return jsonify({"status": "success", "verification_code": code})

@app.route("/api/verify-code", methods=["POST"])
def verify_code():
    data = request.get_json()
    code = data.get("code")
    if not code:
        return jsonify({"status": "error", "message": "Missing code"}), 400
    if session.get("verification_code") != code:
        session["code_attempts"] += 1
        return jsonify({"status": "error", "message": "Invalid code"}), 401
    session["code_verified"] = True
    session.pop("verification_code", None)
    return jsonify({"status": "success", "message": "Code verified"})

@app.route("/api/register", methods=["POST"])
def register_api():
    if not session.get("code_verified"):
        return jsonify({"status": "error", "message": "Verification required"}), 403

    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing user ID"}), 400
    if users_collection.find_one({"user_id": user_id}):
        return jsonify({"status": "error", "message": "User already exists"}), 409

    mfcc_samples = []

    for i in range(1, 11):  # loop over audio1 to audio10
        file_key = f"audio{i}"
        if file_key not in request.files:
            return jsonify({"status": "error", "message": f"Missing {file_key} in form data"}), 400

        audio_file = request.files[file_key]
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
                audio_file.save(temp_webm.name)
                wav_path = temp_webm.name.replace(".webm", ".wav")

            audio = AudioSegment.from_file(temp_webm.name, format="webm")
            audio.export(wav_path, format="wav")

            y, sr = librosa.load(wav_path, sr=None)
            features = extract_mfcc(y, sr)

            if features.shape == (100, 40):
                mfcc_samples.append(features.tolist())
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to process sample {i}: {e}"}), 500
        finally:
            if os.path.exists(temp_webm.name):
                os.remove(temp_webm.name)
            if os.path.exists(wav_path):
                os.remove(wav_path)

    if len(mfcc_samples) < 10:
        return jsonify({"status": "error", "message": "Failed to collect all 10 valid samples."}), 500

    users_collection.insert_one({
        "user_id": user_id,
        "mfcc_samples": mfcc_samples
    })

    # 🔁 Train the CNN model after successful registration
    try:
        model, label_map = train_cnn_model_from_db()
        print("✅ CNN retrained after registration")
    except Exception as e:
        print(f"❌ Failed to train CNN: {e}")

    session.pop("code_verified", None)
    return jsonify({"status": "success", "message": "User registered with 10 samples and CNN retrained."})

@app.route("/api/verify", methods=["POST"])
def verify_user():
    try:
        user_id = request.form.get("user_id")
        audio_file = request.files.get("audio")

        if not user_id or not audio_file:
            return jsonify({"status": "error", "message": "Missing user_id or audio file"}), 400

        # Convert webm to wav
        wav_path = convert_webm_to_wav(audio_file)

        # Load and extract MFCC features
        y, sr = librosa.load(wav_path, sr=None)
        mfcc = extract_mfcc(y, sr)
        mfcc_normalized = normalize_mfcc(mfcc)
        mfcc_input = mfcc_normalized[np.newaxis, ..., np.newaxis]  # shape: (1, 100, 40, 1)

        # Train or load CNN model
        model, label_map = train_cnn_model_from_db()

        # CNN prediction
        prediction = model.predict(mfcc_input)[0]
        predicted_label = np.argmax(prediction)
        predicted_user_id = label_map.get(predicted_label, None)
        confidence = float(np.max(prediction))

        print(f"🔍 CNN Prediction: {predicted_user_id} ({confidence:.2f})")

        # Load reference samples from DB for target user
        user = users_collection.find_one({"user_id": user_id})
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        reference_samples = [np.array(mfcc_sample) for mfcc_sample in user.get("mfcc_samples", [])]

        # Cosine similarity and DTW calculations
        cos_sims = [cosine_similarity(mfcc.reshape(1, -1), ref.reshape(1, -1))[0][0]
                    for ref in reference_samples]
        dtw_distances = [fastdtw(mfcc, ref, dist=euclidean)[0] for ref in reference_samples]

        avg_cos_sim = np.mean(cos_sims)
        avg_dtw = np.mean(dtw_distances)

        print(f"📈 Cosine Similarity Avg: {avg_cos_sim:.3f}")
        print(f"📉 DTW Distance Avg: {avg_dtw:.3f}")

        # Ensemble decision logic (Weighted)
        cnn_score = 1.0 if predicted_user_id == user_id and confidence > 0.85 else 0.0
        cos_score = min(avg_cos_sim / 0.85, 1.0)
        dtw_score = max(1.0 - (avg_dtw / 20000), 0.0)

        weighted_score = cnn_score * 0.5 + cos_score * 0.25 + dtw_score * 0.25

        print(f"🧠 Weighted Score: {weighted_score:.2f} (CNN: {cnn_score:.2f}, Cos: {cos_score:.2f}, DTW: {dtw_score:.2f})")

        result = "success" if weighted_score >= 0.75 else "fail"
        message = "Authentication passed" if result == "success" else "Authentication failed"

        return jsonify({
            "status": result,
            "message": message,
            "cnn_user": predicted_user_id,
            "cnn_confidence": confidence,
            "cosine_similarity": avg_cos_sim,
            "dtw_distance": avg_dtw,
            "weighted_score": weighted_score
        })

    except Exception as e:
        print(f"❌ Verification Error: {e}")
        return jsonify({"status": "error", "message": f"Verification failed: {str(e)}"}), 500

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out"})

@app.route("/api/dashboard", methods=["GET"])
def dashboard_data():
    user = users_collection.find_one({"user_id": session.get("user_id")}, {"_id": 0, "voice_features": 0})
    return jsonify({
        "status": "success",
        "user": user,
        "message": f"Welcome {session.get('user_id')}"
    })

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True)
