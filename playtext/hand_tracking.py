
#cd /Users/mandy/MYTH_Local/playtext
# source venv311/bin/activate
# python hand_tracking.py



import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---- 手势判断函数 ----
def finger_extended(hand_lms, tip_idx, base_idx, h):
    tip_y = hand_lms.landmark[tip_idx].y * h
    base_y = hand_lms.landmark[base_idx].y * h
    return tip_y < base_y

def is_ok_gesture(hand_lms, w, h):
    thumb_tip = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]
    dist = math.hypot((thumb_tip.x - index_tip.x) * w,
                      (thumb_tip.y - index_tip.y) * h)
    if dist < 30:
        middle_ext = finger_extended(hand_lms, 12, 10, h)
        ring_ext = finger_extended(hand_lms, 16, 14, h)
        pinky_ext = finger_extended(hand_lms, 20, 18, h)
        if middle_ext and ring_ext and pinky_ext:
            return True
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
    # 拇指弯曲判断：拇指指尖与食指根部距离
    thumb_tip = hand_lms.landmark[4]
    index_base = hand_lms.landmark[5]
    dist = math.hypot((thumb_tip.x - index_base.x) * w,
                      (thumb_tip.y - index_base.y) * h)
    thumb_bent = dist < 40
    return index_bent and middle_bent and ring_bent and pinky_bent and thumb_bent
def is_rock_gesture(hand_lms, h):
    
    # 食指伸直（指尖高于指根）
    index_ext = finger_extended(hand_lms, 8, 5, h)
    # 小指伸直
    pinky_ext = finger_extended(hand_lms, 20, 17, h)
    # 中指弯曲（指尖低于指根）
    middle_bent = not finger_extended(hand_lms, 12, 9, h)
    # 无名指弯曲
    ring_bent = not finger_extended(hand_lms, 16, 13, h)
    # 拇指状态不影响，可伸可屈，这里不做要求
    return index_ext and pinky_ext and middle_bent and ring_bent
# ---- 摄像头循环 ----
cap = cv2.VideoCapture(0)
print("success, press Q to quit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # 水平镜像
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change the format of colors 

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS) #draw the bullet on hand 
            h, w, _ = img.shape

            # 画食指尖圆点
            idx_tip = hand_lms.landmark[8]
            cx = int(idx_tip.x * w)
            cy = int(idx_tip.y * h)
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), cv2.FILLED)
            # 手势识别
            gesture = "Unknown"
            if is_ok_gesture(hand_lms, w, h):
                gesture = "OK"
            elif is_scissors_gesture(hand_lms, h):
                gesture = "Scissors"
            elif is_rock_gesture(hand_lms, h):
                gesture = "Rock"
            elif is_fist_gesture(hand_lms, h, w):
                gesture = "Fist"

            cv2.putText(img, gesture, (cx, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()