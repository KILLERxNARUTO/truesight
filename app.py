# -----------------------------------------
# Imports
# -----------------------------------------
import os
import time
import random
import string
import numpy as np
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import tensorflow as tf

# -----------------------------------------
# Custom Imports
# -----------------------------------------
from ml_model import (
    get_mfcc_features,
    cosine_similarity,
    dynamic_time_warping,
    load_model,
    trim_silence,
    bandpass_filter,
    train_cnn_model_from_db,
)

# -----------------------------------------
# Environment Setup
# -----------------------------------------
load_dotenv()
os.environ["FFMPEG_PATH"] = os.getenv("FFMPEG_PATH")
os.environ["FFPROBE_PATH"] = os.getenv("FFPROBE_PATH")
default_model_path = os.getenv("MODEL_PATH", "voice_auth_cnn_model.keras")

# -----------------------------------------
# Flask App Initialization
# -----------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")  # Required for session tracking
CORS(app)  # Enable CORS for cross-origin communication

# -----------------------------------------
# MongoDB Setup
# -----------------------------------------
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["voice_auth"]
collection = db["mfcc_features"]

# -----------------------------------------
# Helper: Random Verification Code Generator
# -----------------------------------------
def generate_verification_code(length=6):
    return ''.join(random.choices(string.digits, k=length))

# -----------------------------------------
# Route: Get Verification Code
# -----------------------------------------
@app.route("/api/verification-code", methods=["GET"])
def get_verification_code():
    code = generate_verification_code()
    session["verification_code"] = code
    session["code_created_at"] = int(time.time())  # Track issue time for expiration
    return jsonify({"code": code})

# -----------------------------------------
# Route: Register Voice Samples
# -----------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    user_id = request.form.get("user_id")
    verification_code = request.form.get("verification_code")

    # Check for required fields
    if not user_id or not verification_code:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    # Verify the code
    if verification_code != session.get("verification_code"):
        return jsonify({"status": "error", "message": "Invalid verification code"}), 401

    # Check if user already exists
    if collection.find_one({"user_id": user_id}):
        return jsonify({"status": "error", "message": "User already exists"}), 409

    # Process and store audio files
    files = request.files.getlist("audio")
    if len(files) < 10:
        return jsonify({"status": "error", "message": "At least 10 audio samples required"}), 400

    for idx, file in enumerate(files):
        filename = secure_filename(file.filename)
        wav_path = f"temp_audio_{idx}.wav"

        # Convert WebM to WAV
        audio = AudioSegment.from_file(file, format="webm")
        audio.export(wav_path, format="wav")

        # Preprocess: Filter and trim
        filtered_audio = bandpass_filter(wav_path)
        trimmed_audio = trim_silence(filtered_audio)

        # Extract MFCC
        mfcc = get_mfcc_features(trimmed_audio)
        if mfcc is None:
            continue  # Skip if extraction failed

        # Save to MongoDB
        collection.insert_one({
            "user_id": user_id,
            "mfcc": mfcc.tolist(),
            "code": verification_code
        })

        os.remove(wav_path)  # Clean up

    # Retrain CNN with all DB data
    try:
        model, label_map = train_cnn_model_from_db()
        print("✅ Model retrained after registration")
    except Exception as e:
        print(f"❌ Retraining failed: {e}")

    return jsonify({"status": "success", "message": "Registration successful. Model updated."})

# -----------------------------------------
# Route: Verify User via Voice
# -----------------------------------------
@app.route("/api/verify", methods=["POST"])
def verify():
    user_id = request.form.get("user_id")
    verification_code = request.form.get("verification_code")
    file = request.files.get("audio")

    # Validate input
    if not user_id or not verification_code or not file:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    # Convert input audio to WAV
    filename = secure_filename(file.filename)
    wav_path = f"temp_verify.wav"
    audio = AudioSegment.from_file(file, format="webm")
    audio.export(wav_path, format="wav")

    # Preprocess
    filtered_audio = bandpass_filter(wav_path)
    trimmed_audio = trim_silence(filtered_audio)

    # Extract MFCC
    mfcc = get_mfcc_features(trimmed_audio)
    os.remove(wav_path)
    if mfcc is None:
        return jsonify({"status": "error", "message": "Failed to process audio"}), 500

    # Load trained CNN model
    model, label_map = load_model(default_model_path)

    # Predict class probabilities
    input_data = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(input_data, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    predicted_user = [k for k, v in label_map.items() if v == predicted_index][0]
    confidence = predictions[predicted_index]

    # If CNN predicts different user, reject
    if predicted_user != user_id:
        return jsonify({"status": "error", "message": "Voice mismatch"}), 403

    # Cosine similarity & DTW with stored samples
    user_samples = collection.find({"user_id": user_id})
    similarities, dtw_scores = [], []

    for sample in user_samples:
        stored_mfcc = np.array(sample["mfcc"])
        similarities.append(cosine_similarity(mfcc, stored_mfcc))
        dtw_scores.append(dynamic_time_warping(mfcc, stored_mfcc))

    avg_similarity = np.mean(similarities)
    avg_dtw = np.mean(dtw_scores)

    print(f"✅ Cosine: {avg_similarity:.3f} | DTW: {avg_dtw:.3f} | CNN: {confidence:.3f}")

    # Threshold checks
    if avg_similarity > 0.85 and avg_dtw < 500 and confidence > 0.90:
        return jsonify({
            "status": "success",
            "message": "User authenticated",
            "cosine_similarity": float(avg_similarity),
            "dtw_distance": float(avg_dtw),
            "cnn_confidence": float(confidence),
        })
    else:
        return jsonify({"status": "error", "message": "Authentication failed"}), 401

# -----------------------------------------
# Route: Dashboard (Post Verification Access)
# -----------------------------------------
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    return jsonify({"message": "Welcome to your secure dashboard!"})

# -----------------------------------------
# Route: Logout (Clear Session)
# -----------------------------------------
@app.route("/api/logout", methods=["GET"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})

# -----------------------------------------
# Run the App (if running directly)
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
