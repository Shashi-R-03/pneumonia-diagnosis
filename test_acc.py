import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for the test data

test_dir = 'data/chest_xray/test'  # Path to your test directory

# Create the test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Binary classification (pneumonia vs normal)
)

# Load the trained model
model = load_model('pneumonia_model.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

# Print the test accuracy
print(f"Test Accuracy: {test_acc:.4f}")