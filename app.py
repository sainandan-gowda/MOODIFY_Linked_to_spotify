import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import random
import time
import emoji

# ================== Page Config ==================
st.set_page_config(page_title="Moodify", page_icon=emoji.emojize(":musical_note:"))

# ================== Title & Logo ==================
st.markdown(
    """
    <div style='text-align:center; margin-bottom:20px;'>
        <img src='Gemini_Generated_Image_455dgr455dgr455d.jpg' 
             width='120' 
             style='border-radius:50%; display:block; margin-left:auto; margin-right:auto;'>
        <h1 style='margin-top:10px;'>MOODIFY</h1>
        <h3>A musical mind reader</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Playlist URLs ==================
mood_playlists = {
    "Happy": "https://open.spotify.com/playlist/4B5UJTEyEkofFmBWUXNhBW",
    "Sad": "https://open.spotify.com/playlist/6TyTwY1VKvWmCznL4hHa4R",
    "Neutral": "https://open.spotify.com/playlist/7oleZnEAZBl9JZHQPIrCTt",
    "Surprise": "https://open.spotify.com/playlist/4oV4bV24OtqZGB2ixowB26"
}

# ================== Load Emotion Model ==================
try:
    model = load_model("emotion_cnn.h5")
    st.success(emoji.emojize(":white_check_mark: Emotion model loaded successfully!"))
except:
    st.warning(emoji.emojize(":warning: Model not found. Using random moods."))
    model = None

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
label_map = {
    'Angry': 'Sad',
    'Disgust': 'Surprise',
    'Fear': 'Surprise',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sad': 'Sad',
    'Surprise': 'Surprise'
}

# ================== Emoji + Messages ==================
emoji_map = {
    "Happy": emoji.emojize(":party_popper:"),
    "Sad": emoji.emojize(":cloud_with_rain:"),
    "Neutral": emoji.emojize(":relieved_face:"),
    "Surprise": emoji.emojize(":astonished_face:")
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
    font-size:70px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ================== Webcam Emotion Detection ==================
def detect_emotion_webcam(duration=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return random.choice(list(mood_playlists.keys()))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    collected_emotions = []
    cam_box = st.empty()
    start_time = time.time()

    while time.time() - start_time < duration:
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

            if model:
                preds = model.predict(roi, verbose=0)
                raw_emotion = emotion_labels[np.argmax(preds)]
            else:
                raw_emotion = random.choice(emotion_labels)

            final_emotion = label_map[raw_emotion]
            collected_emotions.append(final_emotion)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, final_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cam_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=300)

    cap.release()
    cam_box.empty()

    if collected_emotions:
        return Counter(collected_emotions).most_common(1)[0][0]
    return random.choice(list(mood_playlists.keys()))

# ================== Detect Mood Button ==================
if st.button("🎬 Detect Mood via Webcam"):
    final_emotion = detect_emotion_webcam()
    playlist_url = mood_playlists[final_emotion]

    st.success(f"🎭 Detected Mood: {final_emotion}")
    st.markdown(f"<div class='emoji-container'>{emoji_map[final_emotion]}</div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;'>{message_map[final_emotion]}</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <a href="{playlist_url}" target="_blank">
            <button style="
                background:#1DB954;
                color:white;
                padding:12px 22px;
                font-size:18px;
                border:none;
                border-radius:10px;
                cursor:pointer;">
                🎧 Open Spotify Playlist
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

# ================== Manual Mood Selection ==================
st.write("### Or select your mood manually:")
cols = st.columns(len(mood_playlists))

for idx, (mood, url) in enumerate(mood_playlists.items()):
    with cols[idx]:
        if st.button(mood):
            st.success(f"🎶 You chose {mood} mode!")
            st.markdown(f"<div class='emoji-container'>{emoji_map[mood]}</div>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center;'>{message_map[mood]}</h3>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <a href="{url}" target="_blank">
                    <button style="
                        background:#1DB954;
                        color:white;
                        padding:10px 18px;
                        font-size:16px;
                        border:none;
                        border-radius:8px;
                        cursor:pointer;">
                        🎶 Play on Spotify
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
