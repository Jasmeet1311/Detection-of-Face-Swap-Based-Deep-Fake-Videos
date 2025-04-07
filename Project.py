import os
import cv2
import tensorflow as tf
import numpy as np
import random
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image  

# Step 1: Set Dataset Path
dataset_path = "dataset"  # Make sure the dataset is extracted here
real_videos_path = os.path.join(dataset_path, "real_videos")
fake_videos_path = os.path.join(dataset_path, "fake_videos")

output_real = "dataset/train/real"
output_fake = "dataset/train/fake"
os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

# Step 2: Extract Frames from Videos
def extract_frames(video_folder, output_folder, frame_rate=10):
    """Extract frames from videos and save them as images."""
    for video in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_rate == 0:
                frame_name = os.path.join(output_folder, f"{video}_frame_{count}.jpg")
                cv2.imwrite(frame_name, frame)
            count += 1
        cap.release()

# Extract frames from real and fake videos
extract_frames(real_videos_path, output_real)
extract_frames(fake_videos_path, output_fake)
print("Frame extraction complete!")

# Step 3: Train the Deepfake Detection Model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output: Real (0) or Fake (1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load Data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory('dataset/train', target_size=(128, 128), batch_size=16, class_mode='binary', subset='training')
val_data = train_datagen.flow_from_directory('dataset/train', target_size=(128, 128), batch_size=16, class_mode='binary', subset='validation')

# Train Model
model = build_model()
model.fit(train_data, validation_data=val_data, epochs=5)
model.save("deepfake_detector.h5")
print("Model training complete!")

# Step 4: Test the Model
model = tf.keras.models.load_model("deepfake_detector.h5")

def predict_random_fake_frame():
    """Automatically picks a random fake image for testing."""
    fake_images = glob.glob("dataset/train/fake/*.jpg")  # Get all fake images
    if not fake_images:
        print("No fake images found! Ensure frame extraction was successful.")
        return
    
    img_path = random.choice(fake_images)  # Pick a random fake image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    print(f"Predicted: {'Fake' if prediction[0][0] > 0.5 else 'Real'} for image {img_path}")

# Run prediction
predict_random_fake_frame()
