# AI Study Assistant - Drowsiness Detection System

Complete implementation of a drowsiness detection system for students with advanced deep learning models and reinforcement learning.

## 📋 Overview

This system uses:
- **Advanced Models**: EfficientNet, MobileNet, ResNet, InceptionV3
- **Hyperparameter Optimization**: Automated grid search
- **Reinforcement Learning**: Learns from user feedback
- **Personalization**: Individual models per student

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook backend/drowsiness_detection_complete.ipynb
```

### 3. Choose Your Path

**Path A: Quick Start (15 minutes)** ⚡ *Recommended*
- Runs cells 1-6 in order
- Trains 1 model with good defaults
- No hyperparameter tuning

**Path B: Full Grid Search (1-2 hours)** 🎯
- Tests all hyperparameter combinations
- Finds optimal parameters
- More accurate but takes longer

## 📂 Project Structure

```
backend/
├── drowsiness_detection_complete.ipynb  ← New organized notebook
├── models.ipynb                         ← Old notebook (can delete)
├── requirements.txt                     ← Dependencies
└── study_assistant_efficientnet_best.h5 ← Trained model (created after running)

dataset_split/
├── train/
│   ├── no yawn/
│   ├── notdrowsy/
│   └── yawn/
└── test/
    ├── no yawn/
    ├── notdrowsy/
    └── yawn/

user_models/          ← Personalized models (auto-created)
feedback_logs/        ← User feedback data (auto-created)
```

## 📖 Notebook Structure

| Step | Cell | Purpose |
|------|------|---------|
| 1 | Imports & Config | Setup environment and paths |
| 2 | Model Builders | Define 4 advanced models |
| 3 | Tuner Class | Hyperparameter grid search class |
| 4 | UserFeedback | Feedback logging system |
| 5 | PersonalizedModel | Per-user model manager |
| 6 | Load Data | Load training and test datasets |
| 7 | **Quick Start** | 🎯 **Choose this for fast results** |
| 8 | Grid Search | Optional: test all combinations |
| 9 | Visualization | Plot results |
| 10 | Demo | Test drowsiness detection |
| 11 | Helpers | Production utility functions |

## 🎯 How It Works

### Training Phase
1. Load dataset (224×224 images)
2. Build pre-trained model (ImageNet weights)
3. Train on drowsiness classification (3 classes)
4. Save trained model

### Inference Phase
1. Capture image from webcam/camera
2. Model predicts: Alert (0) / Distracted (1) / Drowsy (2)
3. If drowsy or random alert → Send break notification
4. User provides feedback: "I'm alert", "I'm drowsy", etc.
5. Model learns from feedback → Fine-tunes weights

## 🏆 Model Comparison

| Model | Speed | Accuracy | RAM | Use Case |
|-------|-------|----------|-----|----------|
| EfficientNet | Fast | ⭐⭐⭐⭐ | Low | Production/Real-time |
| MobileNet | Very Fast | ⭐⭐⭐ | Very Low | Mobile/Edge |
| ResNet | Medium | ⭐⭐⭐⭐ | Medium | General use |
| Inception | Slow | ⭐⭐⭐⭐⭐ | High | Best accuracy |

## 📊 Classes

- **0 - Alert**: Student is attentive, focused
- **1 - Distracted**: Student seems distracted, not concentrating
- **2 - Drowsy**: Student shows drowsiness, yawning, dozing off

## 🔧 Configuration

Edit these in Step 1:

```python
BASE_MODEL_PATH = "study_assistant_model.h5"      # Base model filename
USER_MODELS_DIR = "user_models"                   # Per-user models storage
FEEDBACK_LOG_DIR = "feedback_logs"                # Feedback data storage
CONFIDENCE_THRESHOLD = 0.6                        # Alert if confidence > 60%
```

## 📝 Hyperparameter Options

Edit in Step 8 (Grid Search):

```python
hyperparameter_grid = {
    'learning_rate': [0.001, 0.005, 0.01],      # How fast it learns
    'batch_size': [16, 32],                      # Images per update
    'epochs': [10, 15],                          # Training passes
    'optimizer': ['adam', 'sgd']                 # Optimization algorithm
}
```

## 🎬 Example Usage

### Quick Test
```python
# Initialize for a student
manager = PersonalizedModelManager('student_001')

# Make prediction
prediction = manager.predict('path/to/image.jpg')
print(prediction['label'])  # 'Alert', 'Distracted', or 'Drowsy'

# Get user feedback
manager.process_feedback('path/to/image.jpg', user_feedback=0)  # 0=Alert

# Get statistics
stats = manager.get_user_stats()
print(stats['feedback_stats']['accuracy'])
```

### Batch Processing
```python
feedback_data = [
    {'image_path': 'img1.jpg', 'user_feedback': 0},
    {'image_path': 'img2.jpg', 'user_feedback': 2},
]
stats = batch_process_feedback('student_001', feedback_data)
```

## 🚨 Troubleshooting

### Error: "dataset_split folder not found"
Make sure your data structure matches:
```
dataset_split/
├── train/
│   ├── no yawn/
│   ├── notdrowsy/
│   └── yawn/
└── test/
    ├── no yawn/
    ├── notdrowsy/
    └── yawn/
```

### Error: "final_history not defined"
Make sure you ran cells in order. Run Step 7 (Quick Start) or Step 8 (Grid Search) before Step 9 (Visualization).

### Training is too slow
- Use MobileNet (fastest)
- Reduce batch_size to 16
- Reduce epochs to 10
- Or skip Grid Search and use Quick Start

### Out of memory errors
- Try MobileNet (lowest memory)
- Reduce batch_size from 32 to 16
- Close other applications

## 📈 Performance Tips

1. **Start with Quick Start** (15 min)
2. **If accuracy is low** → Run Grid Search for optimization
3. **If training is slow** → Switch to MobileNet
4. **For best accuracy** → Use InceptionV3

## 📚 Class Paths

**UserFeedback**: Manages feedback logging
- `add_feedback()`: Log correction
- `get_stats()`: Get accuracy metrics
- `get_incorrect_predictions()`: Cases where model was wrong

**PersonalizedModelManager**: Per-user model
- `predict()`: Make prediction
- `process_feedback()`: Log feedback & potentially retrain
- `get_user_stats()`: User statistics

**HyperparameterTuner**: Grid search optimization
- `grid_search()`: Test all combinations
- `get_best_combination()`: Find optimal params
- `save_results()`: Export as CSV

## 🔐 File Outputs

After training:
- `study_assistant_efficientnet_best.h5` - Trained model
- `efficientnet_training_history.png` - Accuracy/Loss curves
- `hyperparameter_results.csv` - Grid search results (if done)
- `user_models/student_001_model.h5` - Student's personalized model
- `feedback_logs/student_001_feedback.json` - Student's feedback history

## 📞 Support

If issues occur:
1. Check dataset structure
2. Verify all imports installed
3. Run cells in order (1→2→3→...)
4. Check cell outputs for error messages

## 📄 License

This project is for educational purposes.

---

**Created**: March 2026
**Version**: 1.0
**Python**: 3.8+
**TensorFlow**: 2.11+
