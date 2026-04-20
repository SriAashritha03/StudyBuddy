import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = r"d:\AI_Study_Assistant\backend\study_assistant_efficientnet_best.h5"
train_dir = r"d:\AI_Study_Assistant\dataset\Train"
test_dir = r"d:\AI_Study_Assistant\dataset\Test"

try:
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    print("\nEvaluating on Training Data...")
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    if train_gen.samples > 0:
        train_loss, train_acc = model.evaluate(train_gen, verbose=0)
    else:
        train_loss, train_acc = 0, 0
        print("No training images found!")
        
    print("\nEvaluating on Test Data...")
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    if test_gen.samples > 0:
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    else:
        test_loss, test_acc = 0, 0
        print("No test images found!")
        
    print("\n--- RESULTS ---")
    print(f"Train Accuracy: {train_acc*100:.2f}% | Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy:  {test_acc*100:.2f}% | Test Loss:  {test_loss:.4f}")
    
    if train_acc < 0.75:
        print("\nCONCLUSION: The model appears to be UNDERFITTING (low training accuracy).")
    elif test_acc < train_acc - 0.1:
        print("\nCONCLUSION: The model appears to be OVERFITTING (good train accuracy, much lower test accuracy).")
    else:
        print("\nCONCLUSION: The model appears to be well-fitted.")

except Exception as e:
    print(f"Error: {e}")
