import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Creating synthetic student dataset...")

# Generate synthetic student data
def create_student_dataset(n_students=2000):
    """
    Create a realistic synthetic dataset of student academic performance
    """
    np.random.seed(42)
    
    # Generate correlated features that make sense for academic performance
    study_hours = np.random.gamma(2, 2, n_students)  # Gamma distribution for study hours
    study_hours = np.clip(study_hours, 0.5, 15)  # Clip to realistic range
    
    # Attendance percentage - somewhat correlated with study hours
    base_attendance = np.random.beta(3, 1, n_students) * 100  # Beta distribution
    attendance_noise = np.random.normal(0, 5, n_students)
    attendance = base_attendance + 0.3 * study_hours + attendance_noise
    attendance = np.clip(attendance, 20, 100)  # Clip to valid percentage range
    
    # Create realistic pass/fail based on study hours and attendance
    # Higher study hours and attendance increase probability of passing
    logit = -3 + 0.4 * study_hours + 0.05 * attendance + np.random.normal(0, 0.5, n_students)
    pass_probability = 1 / (1 + np.exp(-logit))
    passed = np.random.binomial(1, pass_probability, n_students)
    
    # Add some additional realistic features
    previous_gpa = np.random.normal(2.5, 0.8, n_students)
    previous_gpa = np.clip(previous_gpa, 0.0, 4.0)
    
    # Age distribution (18-25 for typical students)
    age = np.random.normal(20, 2, n_students)
    age = np.clip(age, 18, 25).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'study_hours_per_week': np.round(study_hours, 1),
        'attendance_percentage': np.round(attendance, 1),
        'previous_gpa': np.round(previous_gpa, 2),
        'age': age,
        'passed': passed
    })
    
    return df

# Create the dataset
df = create_student_dataset(2000)

# Save to CSV
df.to_csv('student_performance_dataset.csv', index=False)
print(f"Dataset created with {len(df)} students")
print(f"Pass rate: {df['passed'].mean():.2%}")

# Display basic statistics
print("\nDataset Overview:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# Data Visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Distribution of study hours
axes[0, 0].hist(df['study_hours_per_week'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Study Hours per Week', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Study Hours per Week')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Distribution of attendance
axes[0, 1].hist(df['attendance_percentage'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Attendance Percentage', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Attendance Percentage')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Pass/Fail distribution
pass_counts = df['passed'].value_counts()
axes[0, 2].pie([pass_counts[0], pass_counts[1]], labels=['Failed', 'Passed'], 
              colors=['lightcoral', 'lightblue'], autopct='%1.1f%%', startangle=90)
axes[0, 2].set_title('Pass/Fail Distribution', fontsize=14, fontweight='bold')

# Study hours vs Attendance colored by pass/fail
scatter = axes[1, 0].scatter(df['study_hours_per_week'], df['attendance_percentage'], 
                           c=df['passed'], cmap='RdYlBu', alpha=0.6, s=30)
axes[1, 0].set_title('Study Hours vs Attendance (colored by outcome)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Study Hours per Week')
axes[1, 0].set_ylabel('Attendance Percentage')
plt.colorbar(scatter, ax=axes[1, 0], label='Passed (1) / Failed (0)')
axes[1, 0].grid(True, alpha=0.3)

# Box plot for study hours by pass/fail
df.boxplot(column='study_hours_per_week', by='passed', ax=axes[1, 1])
axes[1, 1].set_title('Study Hours Distribution by Outcome', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Outcome (0=Failed, 1=Passed)')
axes[1, 1].set_ylabel('Study Hours per Week')

# Box plot for attendance by pass/fail
df.boxplot(column='attendance_percentage', by='passed', ax=axes[1, 2])
axes[1, 2].set_title('Attendance Distribution by Outcome', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Outcome (0=Failed, 1=Passed)')
axes[1, 2].set_ylabel('Attendance Percentage')

plt.tight_layout()
plt.savefig('data_distribution_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Data distribution plots saved as 'data_distribution_plots.png'")

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[['study_hours_per_week', 'attendance_percentage', 'previous_gpa', 'age', 'passed']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Correlation matrix saved as 'correlation_matrix.png'")

# Prepare data for model training
print("\nPreparing data for model training...")

# Features for the model
feature_columns = ['study_hours_per_week', 'attendance_percentage', 'previous_gpa', 'age']
X = df[feature_columns].values
y = df['passed'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Build the Deep Neural Network
print("\nBuilding Deep Neural Network...")

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train the model with callbacks
print("\nTraining the model...")

# Callbacks for better training
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Model checkpointing - save best model based on validation accuracy
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model_checkpoint.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Model evaluation
print("\nEvaluating model performance...")

# Predictions
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_scaled, y_test, verbose=0)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Performance Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"AUC Score: {auc_score:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Failed', 'Passed']))

# Training history visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training & Validation Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training & Validation Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_and_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Training history and performance plots saved as 'training_history_and_performance.png'")

# Feature importance analysis (using permutation importance approximation)
print("\nAnalyzing feature importance...")

def calculate_feature_importance(model, X, y, feature_names):
    """Calculate feature importance using permutation method"""
    baseline_score = model.evaluate(X, y, verbose=0)[1]  # accuracy
    importances = []
    
    for i in range(X.shape[1]):
        # Make a copy and shuffle one feature
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        
        # Calculate new score
        permuted_score = model.evaluate(X_permuted, y, verbose=0)[1]
        
        # Importance is the decrease in performance
        importance = baseline_score - permuted_score
        importances.append(importance)
    
    return np.array(importances)

importance_scores = calculate_feature_importance(model, X_test_scaled, y_test, feature_columns)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance_scores
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
         color='skyblue', edgecolor='navy', alpha=0.7)
plt.title('Feature Importance (Permutation Method)', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score (Decrease in Accuracy)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Feature importance plot saved as 'feature_importance.png'")

# Prediction examples
print("\nExample Predictions:")
print("-" * 50)

# Create some example students
example_students = np.array([
    [12.0, 95.0, 3.5, 20],  # High study hours, high attendance, good GPA
    [3.0, 65.0, 2.0, 19],   # Low study hours, moderate attendance, poor GPA
    [8.0, 85.0, 3.0, 21],   # Moderate study hours, good attendance, average GPA
    [15.0, 70.0, 2.5, 22],  # Very high study hours, moderate attendance
    [5.0, 95.0, 3.8, 20]    # Low study hours, but high attendance and excellent GPA
])

# Scale the examples
example_students_scaled = scaler.transform(example_students)
predictions = model.predict(example_students_scaled).flatten()

for i, (student, prob) in enumerate(zip(example_students, predictions)):
    outcome = "PASS" if prob > 0.5 else "FAIL"
    print(f"Student {i+1}: Study Hours: {student[0]}, Attendance: {student[1]}%, "
          f"Previous GPA: {student[2]}, Age: {student[3]}")
    print(f"  Prediction: {outcome} (Probability: {prob:.3f})")
    print()

# Save the model
print("Saving the trained model...")
model.save('student_performance_model.h5')
print("Model saved as 'student_performance_model.h5'")

# Load and save the best checkpoint model
print("Loading and saving the best checkpoint model...")
best_model = keras.models.load_model('best_model_checkpoint.h5')

# Evaluate the best model
best_test_loss, best_test_accuracy, best_test_precision, best_test_recall = best_model.evaluate(X_test_scaled, y_test, verbose=0)
best_y_pred_proba = best_model.predict(X_test_scaled).flatten()
best_auc_score = roc_auc_score(y_test, best_y_pred_proba)

print(f"\nBest Model Performance (from checkpoint):")
print(f"Best Test Accuracy: {best_test_accuracy:.4f}")
print(f"Best Test Precision: {best_test_precision:.4f}")
print(f"Best Test Recall: {best_test_recall:.4f}")
print(f"Best AUC Score: {best_auc_score:.4f}")

# Save the scaler for future use
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Create a model performance summary and save to file
performance_summary = f"""
ACADEMIC PERFORMANCE PREDICTION MODEL - DETAILED SUMMARY
========================================================

Dataset Information:
- Total students: {len(df):,}
- Training samples: {len(X_train):,}
- Test samples: {len(X_test):,}
- Pass rate: {df['passed'].mean():.2%}

Features used:
- Study hours per week
- Attendance percentage
- Previous GPA
- Age

Model Architecture:
- Input layer: {X_train_scaled.shape[1]} features
- Hidden layers: 128 → 64 → 32 → 16 neurons (ReLU activation)
- Output layer: 1 neuron (Sigmoid activation)
- Dropout layers: 0.3, 0.3, 0.2 (for regularization)

Final Model Performance:
- Test Accuracy: {test_accuracy:.4f}
- Test Precision: {test_precision:.4f}
- Test Recall: {test_recall:.4f}
- AUC Score: {auc_score:.4f}

Best Checkpoint Model Performance:
- Best Test Accuracy: {best_test_accuracy:.4f}
- Best Test Precision: {best_test_precision:.4f}
- Best Test Recall: {best_test_recall:.4f}
- Best AUC Score: {best_auc_score:.4f}

Training Configuration:
- Optimizer: Adam
- Loss function: Binary crossentropy
- Batch size: 32
- Max epochs: 100
- Early stopping: patience=15 (monitoring val_loss)
- Learning rate reduction: factor=0.2, patience=5

Files Generated:
✓ student_performance_dataset.csv - Generated student dataset
✓ student_performance_model.h5 - Final trained model
✓ best_model_checkpoint.h5 - Best model checkpoint (highest val_accuracy)
✓ scaler.pkl - Fitted StandardScaler
✓ data_distribution_plots.png - Data visualization plots
✓ correlation_matrix.png - Feature correlation heatmap
✓ training_history_and_performance.png - Training curves, confusion matrix, ROC
✓ feature_importance.png - Feature importance analysis
✓ model_summary.txt - This performance summary

Feature Importance Ranking:
"""

# Add feature importance to summary
for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'], feature_importance_df['Importance'])):
    performance_summary += f"{i+1}. {feature}: {importance:.4f}\n"

performance_summary += f"""
Model Usage Instructions:
1. Load the model: model = keras.models.load_model('best_model_checkpoint.h5')
2. Load the scaler: scaler = joblib.load('scaler.pkl')
3. For new predictions:
   - Scale input features: X_new_scaled = scaler.transform(X_new)
   - Predict: predictions = model.predict(X_new_scaled)
   - Probability > 0.5 = Pass, <= 0.5 = Fail

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save performance summary to file
with open('model_summary.txt', 'w') as f:
    f.write(performance_summary)
print("✓ Model performance summary saved as 'model_summary.txt'")

print("\n" + "="*70)
print("ACADEMIC PERFORMANCE PREDICTION MODEL - SUMMARY")
print("="*70)
print(f"✓ Dataset created: {len(df):,} students")
print(f"✓ Model trained with {len(X_train):,} training samples")
print(f"✓ Final model - Test accuracy: {test_accuracy:.3f}, AUC: {auc_score:.3f}")
print(f"✓ Best checkpoint model - Test accuracy: {best_test_accuracy:.3f}, AUC: {best_auc_score:.3f}")
print("\nFiles saved in current directory:")
print("  📊 student_performance_dataset.csv")
print("  🤖 student_performance_model.h5 (final model)")
print("  🏆 best_model_checkpoint.h5 (best model)")
print("  🔧 scaler.pkl")
print("  📈 data_distribution_plots.png")
print("  🔗 correlation_matrix.png") 
print("  📉 training_history_and_performance.png")
print("  📊 feature_importance.png")
print("  📝 model_summary.txt")
print("="*70)