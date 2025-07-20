import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf

# MoveNet 모델 로딩
interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame, size=256):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    h, w = frame.shape[:2]
    for y, x, score in keypoints[0, 0]:
        if score > confidence_threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    return frame

def estimate_pose(frame):
    input_tensor = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

def resize_with_aspect(frame, target_height=360):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(frame, (new_width, target_height))

def add_spacing(frame_left, frame_right, gap=20, bg_color=(0, 0, 0)):
    h1, w1 = frame_left.shape[:2]
    h2, w2 = frame_right.shape[:2]
    target_height = min(h1, h2)
    frame_left_resized = resize_with_aspect(frame_left, target_height)
    frame_right_resized = resize_with_aspect(frame_right, target_height)
    spacing = np.full((target_height, gap, 3), bg_color, dtype=np.uint8)
    return np.hstack((frame_left_resized, spacing, frame_right_resized))

# Streamlit UI
st.title("🏓 탁구 자세 비교 분석 - MoveNet 버전")
st.markdown("사용자와 강사의 자세를 비교합니다. (Powered by TensorFlow MoveNet)")

col1, col2 = st.columns(2)
with col1:
    user_video = st.file_uploader("👤 사용자 영상 업로드", type=["mp4"])
with col2:
    coach_video = st.file_uploader("🎓 강사 영상 업로드", type=["mp4"])

if user_video and coach_video and st.button("📊 자세 비교 시작"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_user:
        tmp_user.write(user_video.read())
        user_path = tmp_user.name
    with tempfile.NamedTemporaryFile(delete=False) as tmp_coach:
        tmp_coach.write(coach_video.read())
        coach_path = tmp_coach.name

    cap_user = cv2.VideoCapture(user_path)
    cap_coach = cv2.VideoCapture(coach_path)
    stframe = st.empty()

    while cap_user.isOpened() and cap_coach.isOpened():
        ret_user, frame_user = cap_user.read()
        ret_coach, frame_coach = cap_coach.read()

        if not ret_user or not ret_coach:
            break

        user_keypoints = estimate_pose(frame_user)
        coach_keypoints = estimate_pose(frame_coach)

        frame_user = draw_keypoints(frame_user, user_keypoints)
        frame_coach = draw_keypoints(frame_coach, coach_keypoints)

        combined = add_spacing(frame_user, frame_coach, gap=30)
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        stframe.image(combined_rgb, caption="사용자 vs 강사", use_column_width=True)

    cap_user.release()
    cap_coach.release()