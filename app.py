import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import random
import time
import emoji

# ================== SAFE OpenCV IMPORT ==================
try:
    import cv2
    OPENCV_AVAILABLE = True
except:
    OPENCV_AVAILABLE = False

# ================== Page Config ==================
st.set_page_config(
    page_title="MOODIFY",
    page_icon=emoji.emojize(":musical_note:"),
    layout="centered"
)

# ================== Title & Logo ==================
st.markdown(
    """
    <div style='text-align:center; margin-bottom:10px;'>
        <img src='Gemini_Generated_Image_455dgr455dgr455d.jpg'
             width='110'
             style='border-radius:50%; display:block; margin-left:auto; margin-right:auto;'>
        <h1 style='margin-top:8px;'>MOODIFY</h1>
        <h4>A musical mind reader</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Spotify Playlists ==================
mood_playlists = {
    "Happy": "https://open.spotify.com/playlist/4B5UJTEyEkofFmBWUXNhBW",
    "Sad": "https://open.spotify.com/playlist/6TyTwY1VKvWmCznL4hHa4R",
    "Neutral": "https://open.spotify.com/playlist/7oleZnEAZBl9JZHQPIrCTt",
    "Surprise": "https://open.spotify.com/playlist/4oV4bV24OtqZGB2ixowB26"
}

# ================== Load Emotion Model (Optional) ==================
try:
    model = load_model("emotion_cnn.h5")
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
label_map = {
    'Angry': 'Sad',
    'Disgust': 'Sad',
    'Fear': 'Sad',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sad': 'Sad',
    'Surprise': 'Surprise'
}

# ================== Emoji & Messages ==================
emoji_map = {
    "Happy": "🎉",
    "Sad": "🌧️",
    "Neutral": "😌",
    "Surprise": "😲"
}

message_map = {
    "Happy": "Keep smiling, the world shines with you 💛",
    "Sad": "It's okay to feel sad, brighter days are ahead 💙",
    "Neutral": "Stay calm and enjoy the moment 😌",
    "Surprise": "Wow! Life is full of surprises ✨"
}

# ================== Styling ==================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #2b2b2b, #1DB954);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    min-height: 100vh;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.emoji-container {
    text-align:center;
    font-size:65px;
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🔼 UI AT THE TOP – MANUAL MOOD SELECTION
# =====================================================
st.markdown("### 🎶 Choose your mood")
cols = st.columns(4)

for col, mood in zip(cols, mood_playlists.keys()):
    with col:
        if st.button(mood):
            st.success(f"🎵 {mood} mode selected")
            st.markdown(f"<div class='emoji-container'>{emoji_map[mood]}</div>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;'>{message_map[mood]}</h4>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <a href="{mood_playlists[mood]}" target="_blank">
                    <button style="
                        background:#1DB954;
                        color:white;
                        padding:10px 18px;
                        font-size:16px;
                        border:none;
                        border-radius:8px;">
                        🎧 Play on Spotify
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

# =====================================================
# 🎬 WEBCAM SECTION (LOGIC UNCHANGED)
# =====================================================
def detect_emotion_webcam(duration=5):
    if not OPENCV_AVAILABLE:
        return random.choice(list(mood_playlists.keys()))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return random.choice(list(mood_playlists.keys()))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    collected = []
    frame_box = st.empty()
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            if MODEL_AVAILABLE:
                preds = model.predict(roi, verbose=0)
                raw = emotion_labels[np.argmax(preds)]
            else:
                raw = random.choice(emotion_labels)

            final = label_map[raw]
            collected.append(final)

        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=280)

    cap.release()
    frame_box.empty()

    if collected:
        return Counter(collected).most_common(1)[0][0]
    return random.choice(list(mood_playlists.keys()))

if OPENCV_AVAILABLE and st.button("🎬 Detect Mood via Webcam"):
    mood = detect_emotion_webcam()
    st.success(f"🎭 Detected Mood: {mood}")
    st.markdown(f"<div class='emoji-container'>{emoji_map[mood]}</div>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>{message_map[mood]}</h4>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <a href="{mood_playlists[mood]}" target="_blank">
            <button style="
                background:#1DB954;
                color:white;
                padding:12px 22px;
                font-size:18px;
                border:none;
                border-radius:10px;">
                🎧 Open Spotify Playlist
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

if not OPENCV_AVAILABLE:
    st.info("📷 Webcam not supported in cloud mode. Please select a mood above.")
