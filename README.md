# DeepFake-Detection-model

A machine learning model designed to detect deepfake videos by analyzing frame-by-frame predictions. The model focuses on achieving generalizability across a variety of deepfake datasets and can be used for both real-time detection and batch processing of video files.


**Features**

**Real-time detection**: Color-coded live predictions for real (green) and fake (red) frames.
**Batch processing**: Analyze entire videos or images in a directory.
**Heatmap Visualization**: Option to display heatmaps for deeper insights into the detection process.
**Graphical output**: Shows prediction trends over time with visual clarity.

**Model Architecture**
The DeepFake-Detection-model employs a hybrid approach, combining convolutional neural networks (CNNs) for spatial feature extraction and temporal modeling for video frames. The architecture is optimized to detect inconsistencies between real and fake frames in videos, leveraging pre-trained networks and custom layers to maximize detection accuracy.

**1. Input:**
Video Frames (224x224, RGB): The input to the model consists of individual video frames resized to 224x224 pixels with three color channels (RGB).

**2. Feature Extraction:**
We use Xception as the backbone for feature extraction. Xception is a widely used architecture in deepfake detection tasks due to its ability to capture subtle facial manipulations.
Pre-trained Xception Network: Initialized with weights pre-trained on ImageNet, the Xception model is particularly effective in learning hierarchical features, from low-level edges and textures to high-level facial structures.

**3. Convolutional Layers (CNN):**
The CNN layers extract spatial features from the input frames:
Conv1: 3x3 filters, stride 1, padding 'same' (128 filters)
Conv2: 3x3 filters, stride 1, padding 'same' (256 filters)
Batch Normalization + ReLU Activation: Helps in stabilizing training and preventing vanishing/exploding gradients.

**4. Temporal Modeling (LSTM/GRU):**
To capture the temporal dependencies between frames, we introduce an LSTM (or GRU, depending on implementation):
LSTM Layer:
Input: Sequence of feature vectors (from CNN) representing multiple frames.
Output: Temporal representation of sequential frames to detect inconsistencies across frames (e.g., sudden pixel manipulations).
Units: 256 (can be tuned).
Bidirectional LSTM (optional): Captures past and future context in frames for better temporal learning.

**6. Classification Head:**
The extracted features are passed through a series of dense layers for final classification:
Dense Layer 1: 512 units, with ReLU activation.
Dropout Layer: Dropout of 0.3 to prevent overfitting.
Dense Layer 2: 256 units, with ReLU activation.
Dropout Layer: Dropout of 0.3.

**Output Layer:**
2 units (Softmax Activation): Predicts the probability of "Real" or "Fake" for each frame.

**7. Loss Function:**
The model uses Categorical Crossentropy Loss for training since it's a binary classification problem, with "Real" and "Fake" as the two classes.

**8. Optimizer:**
Adam Optimizer: Adam is chosen due to its adaptive learning rate, which works well for deep learning tasks.
Learning Rate: 0.001 (with a potential learning rate scheduler to decay it during training).
9. Training:
The model is trained on large-scale deepfake datasets with the following augmentations:

Random Cropping, Flipping, Rotation: To increase the model's robustness to different video manipulations.
Frame Sampling: Randomly samples frames across the video to ensure generalization.
10. Evaluation Metrics:
We evaluate the model using:

**Accuracy**
Precision, Recall, and F1-Score
Confusion Matrix
AUC-ROC Curve: To measure how well the model distinguishes between real and fake frames.




**Deepfake Detection Challenge Dataset**
FaceForensics++
Ensure you have enough computational resources to handle these datasets efficiently.

