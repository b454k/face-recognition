import cv2
import dlib

from utils.face_alignment import align_face

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 for webcam, or provide a video file path
    
    # Load the pre-trained face detector and landmark predictor
    detector = dlib.get_frontal_face_detector() 
    #Only works with frontal faces not from the side or extreme angles
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #This specific model finds 68 landmarks on a face
    """ landmarks:
    0-16	Jawline
    17-21	Left eyebrow
    22-26	Right eyebrow
    27-30	Nose bridge
    30-35	Lower nose
    36-41	Left eye
    42-47	Right eye
    48-59	Outer lip
    60-67	Inner lip"""

    while True:
        success, frame = cap.read() # Read a frame from the webcam 
        # Check if the frame was captured successfully
        if not success:
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            """face.left()	The x-coordinate of the top-left corner
            face.top()	    The y-coordinate of the top-left corner
            face.width()	The width of the rectangle (how wide the face is)
            face.height()	The height of the rectangle (how tall the face is)
            2 is the thickness of the rectangle
            (0, 255, 0) is the color of the rectangle in BGR format"""

            # Get landmarks and align face
            landmarks = predictor(frame, face)
            aligned_face = align_face(frame, landmarks)

            # Display the aligned face
            cv2.imshow("Aligned Face", aligned_face)

        # Display the video frame with detected faces
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #Every 1 ms, the program checks if you pressed the 'q' key
            break
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()