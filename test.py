from tensorflow.keras.models import load_model
model = load_model('pneumonia_model.h5')
model.summary()

