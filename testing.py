import threading
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

pred_label = 'Sign: none'

# labels = {
#     0: 'again',
#     1: 'book',
#     2: 'cat',
#     3: 'color',
#     4: 'dog',
#     5: 'hello',
#     6: 'help',
#     7: 'I',
#     8: 'like',
#     9: 'maybe',
#     10:'play',
#     11: 'stop',
#     12: 'with'
#     }

# labels = {
#     0: 'again',
#     1: 'book',
#     2: 'cat',
#     3: 'color',
#     4: 'hello',
#     5: 'help',
#     6: 'like',
#     7: 'maybe',
#     8:'play',
#     9: 'stop',
#     10: 'with'
#     }


labels = {
    0: 'again',
    1: 'book',
    2: 'hello',
    3: 'help',
    4: 'maybe',
    5:'play',
    6: 'stop',
    7: 'with'
    }

def main():
    dim = (224, 224)
    frames = 10
    channels = 3
    model_path = './saved_models/best_model_300.keras'
    threshold = 0.30
  

    # Define the empty buffer
    frame_buffer = np.empty((0, *dim, channels))

    print("Loading ASL Recognition model")
   
    model = load_model(model_path)

    print("Starting video stream")
    
    cap = cv2.VideoCapture(0)
    # Get the default video FPS and size to ensure compatibility
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output2.avi', fourcc, fps, (frame_width,frame_height))

    def process_frame(frame):
        """Resize and normalize frame."""
        frame_resized = cv2.resize(frame, dim) / 255.0
        return np.reshape(frame_resized, (1, *frame_resized.shape))

    def make_prediction():
        global pred_label

        frame_buffer_reshaped = frame_buffer.reshape(1, *frame_buffer.shape)
        predictions = model.predict(frame_buffer_reshaped)[0]
        best_pred_idx = np.argmax(predictions)
        best_pred_accuracy = predictions[best_pred_idx]
        print(best_pred_idx,best_pred_accuracy)
        if best_pred_accuracy > threshold:
            pred_label = labels[best_pred_idx]
            pred_label = f"Sign: {pred_label}  {best_pred_accuracy * 100:.2f}%"
        else:
            pred_label = "Sign: none"
        print(pred_label)

    # Initialize an empty thread
    x = threading.Thread()
    try:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process and buffer the frame
            frame_processed = process_frame(frame)
            frame_buffer = np.append(frame_buffer, frame_processed, axis=0)

            # Prediction and buffer management
            if frame_buffer.shape[0] == frames:
                if not x.is_alive():
                    x = threading.Thread(target=make_prediction)
                    x.start()

                # Keep the buffer filled with the most recent frames
                frame_buffer = frame_buffer[1:]

            # Display the frame and prediction
            cv2.putText(frame, pred_label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)   
            out.write(frame)     

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
