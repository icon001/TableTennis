import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np

def add_spacing(frame_left, frame_right, gap=20, bg_color=(0, 0, 0)):
    # 높이를 기준으로 정렬
    h1, w1 = frame_left.shape[:2]
    h2, w2 = frame_right.shape[:2]
    target_height = min(h1, h2)

    # 비율 맞춰 리사이즈
    frame_left_resized = resize_with_aspect(frame_left, target_height)
    frame_right_resized = resize_with_aspect(frame_right, target_height)

    # gap에 맞는 빈 영역 생성
    spacing = np.full((target_height, gap, 3), bg_color, dtype=np.uint8)

    # 세 이미지 나란히 합치기
    combined = np.hstack((frame_left_resized, spacing, frame_right_resized))
    return combined

# 비율 유지하면서 프레임 리사이즈 함수
def resize_with_aspect(frame, target_height=360):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(frame, (new_width, target_height))

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

st.title("🏓 탁구 자세 비교 분석")
st.markdown("강사와 사용자의 자세를 나란히 분석하여 비교합니다.")

# 영상 업로드
col1, col2 = st.columns(2)
with col1:
    user_video = st.file_uploader("👤 사용자 영상 업로드", type=["mp4"])
with col2:
    coach_video = st.file_uploader("🎓 강사 영상 업로드", type=["mp4"])

# 분석 버튼
if user_video and coach_video and st.button("📊 자세 비교 시작"):
    # 임시 파일로 저장
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

        # RGB 변환 및 포즈 추적
        user_rgb = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
        coach_rgb = cv2.cvtColor(frame_coach, cv2.COLOR_BGR2RGB)

        user_results = pose.process(user_rgb)
        coach_results = pose.process(coach_rgb)

        # 포즈 랜드마크 표시
        if user_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_user, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if coach_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_coach, coach_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 프레임 정렬
        #frame_user_resized = cv2.resize(frame_user, (480, 360))
        frame_user_resized = resize_with_aspect(frame_user, target_height=360)
        #frame_coach_resized = cv2.resize(frame_coach, (480, 360))
        frame_coach_resized = resize_with_aspect(frame_coach, target_height=360)

        # 높이를 기준으로 맞추고, 가로는 비율 따라 다르게 조절됨
        #combined = np.hstack((frame_user_resized, frame_coach_resized))
        #combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        #stframe.image(combined_rgb, caption="사용자 vs 강사", use_column_width=True)

        combined = add_spacing(frame_user, frame_coach, gap=30)
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        stframe.image(combined_rgb, caption="사용자 vs 강사", use_column_width=True)

    cap_user.release()
    cap_coach.release()
