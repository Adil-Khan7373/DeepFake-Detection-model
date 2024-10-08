import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('deepfake_detection_model.h5')

# Function to draw bounding boxes and annotate predictions
def draw_bounding_boxes(frame, faces, predictions):
    for i, face in enumerate(faces):
        # Extract bounding box from detected face
        x, y, width, height = face['box']
        
        # Annotate prediction (real or fake)
        prediction = predictions[i]
        prediction_text = "Fake" if prediction < 0.5 else "Real"
        confidence = prediction if prediction < 0.5 else 1 - prediction
        
        # Choose the color of the bounding box based on the prediction
        color = (0, 0, 255) if prediction_text == "Fake" else (0, 255, 0)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        # Put text above bounding box
        cv2.putText(frame, f"{prediction_text} ({confidence:.2f})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Function to visualize video with annotations and debug face detection
def visualize_predictions(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()

    total_faces = 0
    total_real = 0
    total_fake = 0
    correct_real = 0
    correct_fake = 0
    real_confidences = []
    fake_confidences = []

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces in the frame
        faces = detector.detect_faces(frame)
        
        # Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected in this frame.")
        else:
            print(f"{len(faces)} face(s) detected in this frame.")
        
        # Extract face regions for prediction if faces are detected
        face_regions = []
        for face in faces:
            x, y, width, height = face['box']
            face_crop = frame[y:y+height, x:x+width]
            face_crop = cv2.resize(face_crop, (224, 224))  
            face_crop = preprocess_input(np.expand_dims(face_crop, axis=0))
            face_regions.append(face_crop)
        
        if face_regions:
            face_regions = np.vstack(face_regions)
            # Make predictions using the model
            predictions = model.predict(face_regions)
            predictions = predictions.flatten()  # Flatten to match predictions with faces

            # Draw bounding boxes and annotate predictions
            annotated_frame = draw_bounding_boxes(frame, faces, predictions)
            
            # Update metrics
            total_faces += len(faces)
            for i, face in enumerate(faces):
                prediction = predictions[i]
                if prediction < 0.5:
                    total_fake += 1
                    fake_confidences.append(prediction)
                    if face.get('label', 0) == 0:  # assume label 0 is fake
                        correct_fake += 1
                else:
                    total_real += 1
                    real_confidences.append(1 - prediction)
                    if face.get('label', 1) == 1:  # assume label 1 is real
                        correct_real += 1

            # Show the frame with annotations
            cv2.imshow("Deepfake Detection", annotated_frame)
            
            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and display metrics
    accuracy = (correct_real + correct_fake) / total_faces
    precision_real = correct_real / total_real if total_real > 0 else 0
    precision_fake = correct_fake / total_fake if total_fake > 0 else 0
    recall_real = correct_real / (correct_real + total_fake - correct_fake) if correct_real + total_fake - correct_fake > 0 else 0
    recall_fake = correct_fake / (correct_fake + total_real - correct_real) if correct_fake + total_real - correct_real > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if precision_real + recall_real > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if precision_fake + recall_fake > 0 else 0

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Real): {precision_real:.2f}")
    print(f"Precision (Fake): {precision_fake:.2f}")
    print(f"Recall (Real): {recall_real:.2f}")
    print(f"Recall (Fake): {recall_fake:.2f}")
    print(f"F1-score (Real): {f1_real:.2f}")
    print(f"F1-score (Fake): {f1_fake:.2f}")

    # Plot confidence distributions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(real_confidences, bins=10, alpha=0.5, label='Real')
    plt.title('Confidence Distribution (Real)')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(fake_confidences, bins=10, alpha=0.5, label='Fake')
    plt.title('Confidence Distribution (Fake)')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage
video_path = r"C:\Users\USER\Desktop\Data set for project\Celeb-real\id47_0000.mp4"
visualize_predictions(video_path)
