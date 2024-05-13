# ASL Dynamic sign Recognition

## Overview
This project utilizes machine learning and computer vision techniques to recognize American Sign Language (ASL) from video input. It is built using TensorFlow, Keras, and OpenCV, and includes both training and prediction components to facilitate the learning and recognition of ASL gestures.

## Requirements
- Python 3.10 or higher
- TensorFlow 2.15.1
- Keras 2.15.1
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yutikachougule/Dynamic-sign-recognition.git
2. Navigate to the project directory:
    ```bash
    cd Dynamic-sign-recognition
3. Install the required Python packages

## File Structure

- data_utils.py: Utility functions for processing labels and videos.
- frame_generator.py: Custom data generator for video frame processing.
- defined_model.py: Contains the definition of the ASL recognition model.
- data/: Directory containing training, validation, and test video datasets. Add the "data" folder before running the code.
- saved_models/: Directory where trained model weights are saved.
- plots/: Directory where training plots such as loss and accuracy graphs are stored.
- training.py: Script for training the dynamic sign recognition model.
- testing.py: Script for real-time dynamic sign prediction using a webcam

## Dataset 
Add your data folder before starting the training. <br />
The /data folder is of format: <br />
* data
    * Training
        * 1.mp4
        * 2.mp4
        * ....
    * Testing
        * 1.mp4
        * 2.mp4
        * ....
    * Validation
        * 1.mp4
        * 2.mp4
        * ....
## Training the model
To train the ASL recognition model, run:

```bash
python training.py
```

This script trains the model using video data stored in data/, saves the model's weights in saved_models/, and generates training plots in plots/.


## Real - time prediction
To start real-time gesture recognition, run:

```bash
python testing.py
```

Ensure you have a webcam connected to your system. This script uses the trained model to predict ASL signs in real-time from the webcam input.




