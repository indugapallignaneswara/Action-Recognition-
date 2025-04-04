import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from utils.utils import *
from classifier.model import *
from sklearn.preprocessing import LabelEncoder

# Constants (can be moved to a config)
MAX_SEQ_LENGTH = 32
NUM_FEATURES = 1024
INPUT_SIZE = (224, 224)
NUM_CLASSES = 5
DENSE_DIM = 512
NUM_HEADS = 2

feature_extractor = build_densenet121_feature_extractor()

# Load the trained transformer model (with custom layers)
def load_trained_transformer_model(model_path="classifier/video_classifier_2025-04-04_12-13-05.weights.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    input_shape = (MAX_SEQ_LENGTH, NUM_FEATURES)
    model = get_compiled_model(
        input_shape=input_shape,
        sequence_length=MAX_SEQ_LENGTH,
        embed_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        dense_dim=DENSE_DIM,
        num_heads=NUM_HEADS,
    )
    model.load_weights(model_path)
    return model

# # Predict the action class from a single video
# def predict_action(video_path, model, feature_extractor):
#     frames = load_single_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=INPUT_SIZE)

#     features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
#     for i in range(len(frames)):
#         if np.mean(frames[i]) > 0.0:
#             features[i] = feature_extractor.predict(frames[None, i])[0]
#         else:
#             features[i] = 0.0

#     # Add batch dimension
#     features = np.expand_dims(features, axis=0)  # shape: (1, MAX_SEQ_LENGTH, NUM_FEATURES)

#     prediction = model.predict(features)[0]
#     predicted_class = int(np.argmax(prediction))
#     confidence = float(np.max(prediction))

#     return predicted_class, confidence
def predict_action(video_path, trained_model, label_processor, max_frames=32, num_features=1024):
    """
    Loads a video, preprocesses it, predicts the class using the trained model, and displays the frames as a GIF.

    Args:
    - video_path (str): Path to the video file.
    - trained_model (keras.Model): Trained video classification model.
    - label_processor (LabelEncoder): Label encoder used to convert class labels to integers.
    - max_frames (int): Maximum number of frames to load.
    - num_features (int): Number of features extracted by the feature extractor.

    Returns:
    - predicted_label (str): The predicted class label.
    """
    # Load and preprocess the video
    frames = load_single_video(video_path, max_frames=max_frames)

    # 🛠 Debug print to ensure shape of frames
    print(f"📦 Loaded frames shape: {frames.shape}")  # Expecting: (max_frames, 224, 224, 3)

    # Extract features using the pretrained feature extractor
    temp_features = np.zeros((max_frames, num_features), dtype="float32")

    for i in range(len(frames)):
        frame = frames[i]  # shape: (224, 224, 3)

        # 🧠 Print shape of the individual frame
        print(f"🔍 Frame {i} shape: {frame.shape} — mean pixel value: {np.mean(frame):.4f}")

        if np.mean(frame) > 0.0:
            single_frame = np.expand_dims(frame, axis=0)  # shape: (1, 224, 224, 3)

            print(f"🚀 Running prediction for frame {i}...")

            try:
                feature_vec = feature_extractor.predict(single_frame)
                print(f"✅ Frame {i} extracted feature shape: {feature_vec.shape}")

                temp_features[i, :] = feature_vec[0]  # assign to (1024,)
            except Exception as e:
                print(f"❌ Error predicting frame {i}: {e}")
                temp_features[i, :] = 0.0
        else:
            print(f"⚠️ Frame {i} is blank or empty, skipping feature extraction.")
            temp_features[i, :] = 0.0


    # Add batch dimension (1, max_frames, num_features) for prediction
    temp_features = np.expand_dims(temp_features, axis=0)

    label_processor = LabelEncoder()
    label_processor.classes_ = np.array(['CricketShot', 'PlayingCello', 'Punch', 'ShavingBeard', 'TennisSwing'])


    # Predict the class for the video
    predictions = trained_model.predict(temp_features)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Convert predicted integer back to the original class label
    predicted_label = label_processor.inverse_transform(predicted_class_index)

    return predicted_label[0]

# def predict_action(video_path, trained_model, label_processor, max_frames=32, num_features=1024):
#     """
#     Loads a video, extracts features using a pretrained CNN, and classifies the action using a transformer model.

#     Returns:
#         predicted_label (str): The predicted class label.
#     """
#     # Load and preprocess the video
#     frames = load_single_video(video_path, max_frames=max_frames)
#     print(f"\n📦 Loaded video: {video_path}")
#     print(f"📏 Frames shape: {frames.shape}")  # Expecting: (max_frames, 224, 224, 3)

#     # Initialize feature array
#     temp_features = np.zeros((max_frames, num_features), dtype="float32")

#     for i in range(max_frames):
#         frame = frames[i]  # ✅ This is safe now

#         if np.mean(frame) > 0.0:
#             single_frame = np.expand_dims(frame, axis=0)  # shape: (1, 224, 224, 3)
#             feature_vec = feature_extractor.predict(single_frame)
#             temp_features[i, :] = feature_vec[0]
#         else:
#             temp_features[i, :] = 0.0

#     print("✅ frames is numpy array:", isinstance(frames, np.ndarray))
#     print("frames dtype:", frames.dtype)
#     print("frames shape:", frames.shape)
#     print("temp_features shape:", temp_features.shape)


#     # Add batch dimension
#     temp_features = np.expand_dims(temp_features, axis=0)  # Shape: (1, max_frames, num_features)

#     print("🔮 Running final prediction on feature sequence...")
#     try:
#         predictions = trained_model.predict(temp_features)
#         predicted_class_index = np.argmax(predictions, axis=1)[0]
#         predicted_label = label_processor.inverse_transform([predicted_class_index])[0]
#         print(f"🎯 Predicted class: {predicted_label} (confidence: {np.max(predictions):.4f})")
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")
#         return "Unknown"

#     return predicted_label