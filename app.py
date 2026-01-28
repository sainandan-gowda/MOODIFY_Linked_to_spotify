import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import random
import time
import emoji

# ==================  OpenCV IMPORT ==================
try:
    import cv2
    OPENCV_AVAILABLE = True
except:
    OPENCV_AVAILABLE = False

# ================== SPEECH IMPORT ==================
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except:
    SPEECH_AVAILABLE = False


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

# ================== LOAD CNN MODEL (ONLY FOR LOCAL SYSTEM)==================
MODEL_AVAILABLE = False
model = None

if OPENCV_AVAILABLE:
    try:
        model = load_model("emotion_cnn.h5")
        MODEL_AVAILABLE = True
        st.success(emoji.emojize(":white_check_mark: CNN emotion model loaded"))
    except:
        st.error(
            emoji.emojize(
                ":x: emotion_cnn.h5 not found. "
                "Local execution REQUIRES this file."
            )
        )
        st.stop()

# ================== Emotion Labels ==================
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

# ================== Cloud Warnings ==================
if not OPENCV_AVAILABLE:
    st.warning(
        emoji.emojize(
            ":camera: Webcam & voice are disabled in cloud mode. "
            "Please select your mood manually."
        )
    )

# ================== Emojis & Messages ==================
emoji_map = {
    "Happy": emoji.emojize(":party_popper:"),
    "Sad": emoji.emojize(":cloud_with_rain:"),
    "Neutral": emoji.emojize(":relieved_face:"),
    "Surprise": emoji.emojize(":astonished_face:")
}

message_map = {
    "Happy": emoji.emojize("Keep smiling, the world shines with you :yellow_heart:"),
    "Sad": emoji.emojize("It's okay to feel sad, brighter days are ahead :blue_heart:"),
    "Neutral": emoji.emojize("Stay calm and enjoy the moment :relieved_face:"),
    "Surprise": emoji.emojize("Wow! Life is full of surprises :sparkles:")
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
# 🔼 MANUAL MOOD SELECTION
# =====================================================
st.markdown(emoji.emojize("### :musical_note: Choose your mood"))
cols = st.columns(4)

for col, mood in zip(cols, mood_playlists.keys()):
    with col:
        if st.button(mood):
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
                        {emoji.emojize(":headphone: Play on Spotify")}
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

# =====================================================
# 🎬 WEBCAM (CNN MODEL USED)
# =====================================================
def detect_emotion_webcam(duration=5):
    cap = cv2.VideoCapture(0)
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

            preds = model.predict(roi, verbose=0)
            raw = emotion_labels[np.argmax(preds)]
            collected.append(label_map[raw])

        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=280)

    cap.release()
    frame_box.empty()
    return Counter(collected).most_common(1)[0][0]

if OPENCV_AVAILABLE and st.button(emoji.emojize(":movie_camera: Detect Mood via Webcam")):
    mood = detect_emotion_webcam()
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
                {emoji.emojize(":headphone: Open Spotify Playlist")}
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# 🎙️ VOICE MOOD DETECTION (LOCAL ONLY)
# =====================================================
def detect_mood_from_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... say happy, sad, neutral or surprise")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, phrase_time_limit=4)

    try:
        text = r.recognize_google(audio).lower()
        st.success(f"You said: {text}")
        for mood in mood_playlists:
            if mood.lower() in text:
                return mood
    except:
        st.error("Voice not recognized")
    return None

if OPENCV_AVAILABLE and SPEECH_AVAILABLE and st.button(emoji.emojize(":microphone: Detect Mood via Voice")):
    mood = detect_mood_from_voice()
    if mood:
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
                    {emoji.emojize(":headphone: Open Spotify Playlist")}
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
st.warning(
    emoji.emojize(
        ":information_source: Face and voice-based emotion detection are disabled in cloud deployment "
        "due to hardware access limitations."
    )
)
