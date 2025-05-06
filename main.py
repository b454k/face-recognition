import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Global variables for trackbars
scaling_factor = 2.0
vertical_shift = 0.0

def on_scaling_change(val):
    global scaling_factor
    scaling_factor = val / 10  # Scale trackbar value to a float

def on_vertical_shift_change(val):
    global vertical_shift
    vertical_shift = (val - 50) / 100  # Center trackbar value at 0

def detect_and_process_face(image, crop_size=256):
    global scaling_factor, vertical_shift

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect facial landmarks
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None

    # Get the first detected face
    face_landmarks = results.multi_face_landmarks[0]

    # Extract eye landmarks
    left_eye = np.mean([[lm.x, lm.y] for lm in [face_landmarks.landmark[i] for i in [33, 133]]], axis=0)
    right_eye = np.mean([[lm.x, lm.y] for lm in [face_landmarks.landmark[i] for i in [362, 263]]], axis=0)

    # Convert normalized coordinates to pixel coordinates
    h, w, _ = image.shape
    left_eye = np.array([left_eye[0] * w, left_eye[1] * h])
    right_eye = np.array([right_eye[0] * w, right_eye[1] * h])

    # Align the face by rotating the image
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))
    rotation_matrix = cv2.getRotationMatrix2D((float(left_eye[0]), float(left_eye[1])), angle, 1)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Transform landmarks to the aligned image
    def transform_landmark(landmark):
        x, y = landmark.x * w, landmark.y * h
        transformed = np.dot(rotation_matrix, np.array([x, y, 1]))
        return transformed[:2]

    transformed_landmarks = [transform_landmark(lm) for lm in face_landmarks.landmark]
    transformed_landmarks = np.array(transformed_landmarks)

    # Calculate face bounding box
    x_min, y_min = np.min(transformed_landmarks, axis=0)
    x_max, y_max = np.max(transformed_landmarks, axis=0)
    face_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    face_size = max(x_max - x_min, y_max - y_min)

    # Dynamically scale the crop size based on the face size
    dynamic_crop_size = int(face_size * scaling_factor)

    # Use the dynamically scaled crop size around the face center
    crop_center = face_center + np.array([0, vertical_shift * dynamic_crop_size])
    x1 = int(crop_center[0] - dynamic_crop_size / 2)
    y1 = int(crop_center[1] - dynamic_crop_size / 2)
    x2 = x1 + dynamic_crop_size
    y2 = y1 + dynamic_crop_size

    # Ensure the crop is within image bounds
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    # Crop the face
    cropped_face = aligned_image[y1:y2, x1:x2]

    # Resize to ensure consistent output size
    cropped_face = cv2.resize(cropped_face, (crop_size, crop_size))

    return cropped_face

def process_webcam_feed(crop_size=256):
    global scaling_factor, vertical_shift

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create OpenCV trackbars for interactive controls
    cv2.namedWindow("Controls")
    cv2.createTrackbar("Scaling", "Controls", int(scaling_factor * 10), 50, on_scaling_change)
    cv2.createTrackbar("Vertical Shift", "Controls", 50, 100, on_vertical_shift_change)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame
        cropped_face = detect_and_process_face(frame, crop_size)

        # Display the results
        if cropped_face is not None:
            combined_view = np.hstack((cv2.resize(frame, (crop_size, crop_size)), cropped_face))
            cv2.imshow("Original and Cropped Face", combined_view)
        else:
            cv2.imshow("Original and Cropped Face", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam_feed(crop_size=256)