import cv2
import mediapipe as mp
import math
import pygame
import time

# ========== 初始化音频 ==========
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
bass_sound = pygame.mixer.Sound("bass.mp3")
vocal_sound = pygame.mixer.Sound("vocal.mp3")
drums_sound = pygame.mixer.Sound("drums.mp3")
other_sound = pygame.mixer.Sound("other.mp3")

tracks = {
    "bass": bass_sound,
    "vocal": vocal_sound,
    "drums": drums_sound,
    "other": other_sound
}

music_started = False
track_active = {"bass": False, "vocal": False, "drums": False, "other": True}

def start_all_tracks():
    global music_started
    if music_started:
        return
    for name, sound in tracks.items():
        sound.play(-1)
        if name == "other":
            sound.set_volume(1.0)
        else:
            sound.set_volume(0.0)
    music_started = True
    print("🎵 All tracks started. Background (other) playing.")

def mute_all():
    for sound in tracks.values():
        sound.set_volume(0.0)
    print("🔇 All tracks muted.")

def unmute_track(track_name):
    if track_name in tracks and track_name != "other":
        tracks[track_name].set_volume(1.0)
        if not track_active[track_name]:
            track_active[track_name] = True
            print(f"🔊 {track_name} activated.")
        else:
            # 仍然打印但只会在手势改变时调用，所以不会反复
            print(f"🔊 {track_name} (already active)")

def reset_after_fist():
    tracks["other"].set_volume(1.0)
    for name in ["bass", "vocal", "drums"]:
        tracks[name].set_volume(0.0)
        track_active[name] = False
    print("🔄 After fist: other resumes, others muted and reset.")

# ========== MediaPipe 初始化 ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ========== 辅助函数 ==========
def finger_extended(hand_lms, tip_idx, base_idx, h):
    tip_y = hand_lms.landmark[tip_idx].y * h
    base_y = hand_lms.landmark[base_idx].y * h
    return tip_y < base_y

def thumb_extended_by_x(hand_lms, w):
    thumb_tip_x = hand_lms.landmark[4].x * w
    index_tip_x = hand_lms.landmark[8].x * w
    diff = thumb_tip_x - index_tip_x
    return diff > 30

def thumb_bent_by_x(hand_lms, w):
    thumb_tip_x = hand_lms.landmark[4].x * w
    index_tip_x = hand_lms.landmark[8].x * w
    diff = thumb_tip_x - index_tip_x
    return diff < 20

# ========== 手势定义 ==========
def is_ok_gesture(hand_lms, w, h):
    thumb_tip = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]
    dist = math.hypot((thumb_tip.x - index_tip.x) * w,
                      (thumb_tip.y - index_tip.y) * h)
    if dist < 30:
        middle_ext = finger_extended(hand_lms, 12, 10, h)
        ring_ext = finger_extended(hand_lms, 16, 14, h)
        pinky_ext = finger_extended(hand_lms, 20, 18, h)
        return middle_ext and ring_ext and pinky_ext
    return False

def is_scissors_gesture(hand_lms, h):
    index_ext = finger_extended(hand_lms, 8, 5, h)
    middle_ext = finger_extended(hand_lms, 12, 9, h)
    ring_bent = not finger_extended(hand_lms, 16, 13, h)
    pinky_bent = not finger_extended(hand_lms, 20, 17, h)
    return index_ext and middle_ext and ring_bent and pinky_bent

def is_fist_gesture(hand_lms, h, w):
    index_bent = not finger_extended(hand_lms, 8, 5, h)
    middle_bent = not finger_extended(hand_lms, 12, 9, h)
    ring_bent = not finger_extended(hand_lms, 16, 13, h)
    pinky_bent = not finger_extended(hand_lms, 20, 17, h)
    thumb_bent_flag = thumb_bent_by_x(hand_lms, w)
    return index_bent and middle_bent and ring_bent and pinky_bent and thumb_bent_flag

def is_rock_gesture(hand_lms, h, w):
    index_ext = finger_extended(hand_lms, 8, 5, h)
    pinky_ext = finger_extended(hand_lms, 20, 17, h)
    middle_bent = not finger_extended(hand_lms, 12, 9, h)
    ring_bent = not finger_extended(hand_lms, 16, 13, h)
    thumb_ext = thumb_extended_by_x(hand_lms, w)
    return index_ext and pinky_ext and middle_bent and ring_bent and thumb_ext

def is_aki_gesture(hand_lms, h, w):
    index_ext = finger_extended(hand_lms, 8, 5, h)
    pinky_ext = finger_extended(hand_lms, 20, 17, h)
    middle_bent = not finger_extended(hand_lms, 12, 9, h)
    ring_bent = not finger_extended(hand_lms, 16, 13, h)
    thumb_bent_flag = thumb_bent_by_x(hand_lms, w)
    return index_ext and pinky_ext and middle_bent and ring_bent and thumb_bent_flag

# ========== 主循环 ==========
cap = cv2.VideoCapture(0)
print("🎧 DJ mode activated. Press 'q' to exit.")
print("Gesture mapping: Rock → drums | Aki → bass | OK → vocal | Fist (held) → mute all")

last_gesture = None   # 记录上一次识别的手势（不包括拳头）
fist_active = False

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        h, w, _ = img.shape

        idx_tip = hand_lms.landmark[8]
        cx = int(idx_tip.x * w)
        cy = int(idx_tip.y * h)

        if not music_started:
            start_all_tracks()

        # 识别手势
        gesture = "Unknown"
        current_gesture = None

        if is_fist_gesture(hand_lms, h, w):
            gesture = "Fist ✊"
            current_gesture = "mute"
        elif is_rock_gesture(hand_lms, h, w):
            gesture = "Rock 🤘"
            current_gesture = "drums"
        elif is_aki_gesture(hand_lms, h, w):
            gesture = "Aki 🖖"
            current_gesture = "bass"
        elif is_ok_gesture(hand_lms, w, h):
            gesture = "OK 👌"
            current_gesture = "vocal"

        # 拳头静音逻辑
        if current_gesture == "mute":
            if not fist_active:
                mute_all()
                fist_active = True
            # 拳头状态下，last_gesture 保持不变，不更新
        else:
            if fist_active:
                reset_after_fist()
                fist_active = False
                # 拳头松开后，清除 last_gesture，以便下次手势能激活
                last_gesture = None
            # 非拳头手势：仅当手势改变时才激活音轨
            if current_gesture and current_gesture != last_gesture:
                unmute_track(current_gesture)
                last_gesture = current_gesture

        # 显示文本
        if current_gesture == "mute":
            display_text = "Fist -- Mute All"
        elif current_gesture == "drums":
            display_text = "Rock -- drums"
        elif current_gesture == "bass":
            display_text = "Aki -- bass"
        elif current_gesture == "vocal":
            display_text = "OK -- vocal"
        else:
            display_text = gesture

        cv2.putText(img, display_text, (cx, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), cv2.FILLED)

    else:
        # 没有检测到手：可保持 last_gesture 不变，不影响逻辑
        pass

    cv2.imshow("Hand Gesture DJ", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for sound in tracks.values():
    sound.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()