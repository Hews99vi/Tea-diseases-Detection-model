import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Load the model
model_path = 'trained_model.keras'
model = tf.keras.models.load_model(model_path)

# Print model summary
model.summary()

# Define class names
class_names = [
    'Anthracnose',
    'algal leaf',
    'bird eye spot',
    'brown blight',
    'gray light',
    'healthy',
    'red leaf spot',
    'white spot'
]

# Create a test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_dir = 'test'
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
start_time = time.time()
test_loss, test_accuracy = model.evaluate(test_generator)
evaluation_time = time.time() - start_time

print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Evaluation time: {evaluation_time:.2f} seconds")

# Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save results to file
results = {
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'evaluation_time': float(evaluation_time),
    'classification_report': report
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Plot classification report as a heatmap
plt.figure(figsize=(12, 8))
report_df_plot = report_df.iloc[:-3, :3]  # Exclude avg rows and only include precision, recall, f1-score
sns.heatmap(report_df_plot.astype(float), annot=True, cmap='YlGnBu')
plt.title('Classification Report')
plt.tight_layout()
plt.savefig('classification_report.png')

print("\nEvaluation results saved to 'evaluation_results.json'")
print("Confusion matrix saved to 'confusion_matrix.png'")
print("Classification report saved to 'classification_report.png'")

# Plot some example predictions
def plot_sample_predictions(num_samples=5):
    plt.figure(figsize=(15, 10))
    
    # Reset the generator to the beginning
    test_generator.reset()
    
    # Get a batch of images and their true labels
    images, labels = next(test_generator)
    
    # Make predictions
    preds = model.predict(images)
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i])
        
        true_class = np.argmax(labels[i])
        pred_class = np.argmax(preds[i])
        
        title_color = 'green' if true_class == pred_class else 'red'
        
        plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}", 
                 color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("Sample predictions saved to 'sample_predictions.png'")

plot_sample_predictions()
