import cv2
import mediapipe as mp

mph= mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
hands = mph.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2)

# Define mapping from hand landmarks to sign language words or phrases
landmark_to_word = {
    mph.HandLandmark.WRIST: " ",
    mph.HandLandmark.THUMB_TIP: "A",
    mph.HandLandmark.THUMB_IP: "B",
    mph.HandLandmark.THUMB_MCP: "C",
    mph.HandLandmark.INDEX_FINGER_TIP: "D",
    mph.HandLandmark.INDEX_FINGER_DIP: "E",
    mph.HandLandmark.INDEX_FINGER_PIP: "F",
    mph.HandLandmark.INDEX_FINGER_MCP: "G",
    mph.HandLandmark.MIDDLE_FINGER_TIP: "H",
    mph.HandLandmark.MIDDLE_FINGER_DIP: "I",
    mph.HandLandmark.MIDDLE_FINGER_PIP: "J",
    mph.HandLandmark.MIDDLE_FINGER_MCP: "K",
    mph.HandLandmark.RING_FINGER_TIP: "L",
    mph.HandLandmark.RING_FINGER_DIP: "M",
    mph.HandLandmark.RING_FINGER_PIP: "N",
    mph.HandLandmark.RING_FINGER_MCP: "O",
    mph.HandLandmark.PINKY_TIP: "P",
    mph.HandLandmark.PINKY_DIP: "Q",
    mph.HandLandmark.PINKY_PIP: "R",
    mph.HandLandmark.PINKY_MCP: "S",
    mph.HandLandmark.THUMB_CMC: "T",
    mph.HandLandmark.INDEX_FINGER_MCP: "U",
    mph.HandLandmark.MIDDLE_FINGER_MCP: "V",
    mph.HandLandmark.RING_FINGER_MCP: "W",
    mph.HandLandmark.PINKY_MCP: "X",
    mph.HandLandmark.INDEX_FINGER_TIP: "Y",
    mph.HandLandmark.MIDDLE_FINGER_TIP: "Z",
    mph.HandLandmark.THUMB_CMC: "Thumb CMC",
    mph.HandLandmark.INDEX_FINGER_MCP: "Index MCP",
    mph.HandLandmark.MIDDLE_FINGER_MCP: "Middle MCP",
}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Get hand landmarks
    hand_gesture = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:          
            for lm in mph.HandLandmark:
                if lm in landmark_to_word and hand_landmarks.landmark[lm].visibility > 0.2:
                    hand_gesture += landmark_to_word[lm]
           
            # Display the resulting image with hand landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, hand_landmarks, mph.HAND_CONNECTIONS)

    if hand_gesture:
        print(hand_gesture)
        cv2.putText(image, hand_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('MediaPipe Hands', image)

    # Exit on ESC key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
