import numpy as np
import cv2

def align_face(frame, landmarks):
    # Convert dlib landmarks to a NumPy array
    points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

    """ [[x0, y0],
        [x1, y1],
        [x2, y2],
         ...
        [x67, y67]]"""


    # Get the coordinates of the eyes (landmarks 36-39 for the left eye, 42-45 for the right eye)
    left_eye = np.mean(points[36:42], axis=0)  # Average of left eye landmarks
    right_eye = np.mean(points[42:48], axis=0)  # Average of right eye landmarks

    # Calculate the angle of rotation
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    # arctan2: When x=1, y=1 vector (1, 1) is 45° from the x-axis. in radians : 0.785398



    # Get the center point between the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    """[[ cos(θ)*s, -sin(θ)*s, tx],
         [ sin(θ)*s,  cos(θ)*s, ty]]"""

    # Rotate this frame using rotation_matrix, and output an image with the same width and height as the original.
    aligned_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

    # Crop the aligned face using the bounding rectangle
    x, y, w, h = cv2.boundingRect(points)
    cropped_face = aligned_frame[y:y + h, x:x + w]

    return cropped_face