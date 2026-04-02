#!/usr/bin/env python3.14
"""
Train emotion detection model on FER2013 dataset
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile

# Download FER2013 dataset (if not already present)
def download_fer2013():
    print("📥 Downloading FER2013 dataset...")
    
    # Note: FER2013 requires registration, but you can use alternative sources
    # For now, we'll create a simple dataset builder
    
    data_file = "fer2013.csv"
    if not os.path.exists(data_file):
        print("⚠️  Please download fer2013.csv from:")
        print("https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        print("And place it in the current directory")
        return False
    return True

def build_model():
    """Build CNN model for emotion detection"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_fer2013_data():
    """Load FER2013 dataset"""
    print("📂 Loading FER2013 dataset...")
    
    try:
        data = pd.read_csv('fer2013.csv')
        print(f"✓ Loaded {len(data)} samples")
        
        # Parse pixel data
        X = []
        y = []
        
        for idx, row in data.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48, 1)
            X.append(pixels / 255.0)
            y.append(row['emotion'])
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} samples...")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Data shape: {X.shape}")
        return X, y
    
    except FileNotFoundError:
        print("❌ fer2013.csv not found!")
        return None, None

def main():
    print("\n" + "="*60)
    print("🚀 EMOTION MODEL TRAINER")
    print("="*60 + "\n")
    
    # Check dataset
    if not download_fer2013():
        print("\n⚠️  Dataset not available. Skipping training.")
        return
    
    # Load data
    X, y = load_fer2013_data()
    if X is None:
        return
    
    # Convert labels to one-hot
    y = keras.utils.to_categorical(y, 7)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Build model
    print("\n🔨 Building model...")
    model = build_model()
    model.summary()
    
    # Train model
    print("\n🎓 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    model.save('emotion_model.hdf5')
    print("\n✓ Model saved as 'emotion_model.hdf5'")

if __name__ == "__main__":
    main()