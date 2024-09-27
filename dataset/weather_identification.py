import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import tensorflow as tf  
from tensorflow.keras.applications import VGG16  
from tensorflow.keras import layers, models  
import matplotlib.pyplot as plt  

img_height, img_width = 250, 250
batch_size = 32  
num_classes = 4  

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(  
    "E:\\Weather Identification\\dataset",  
    image_size=(img_height, img_width),  
    batch_size=batch_size,  
    label_mode='categorical',  
    seed=42,  
    subset="training",  
    validation_split=0.3 
)  

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(  
    "E:\\Weather Identification\\dataset",  
    image_size=(img_height, img_width),  
    batch_size=batch_size,  
    label_mode='categorical',  
    seed=42,  
    subset="validation",  
    validation_split=0.3  
)  

base_model = VGG16(weights='imagenet', include_top=False)  

for layer in base_model.layers:  
    layer.trainable = False  

model = models.Sequential()  
model.add(base_model)  
model.add(layers.GlobalAveragePooling2D())  
model.add(layers.Dense(128, activation='relu'))  
model.add(layers.Dropout(0.5))  
model.add(layers.Dense(num_classes, activation='softmax'))  

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

history = model.fit(  
    train_dataset,  
    validation_data=validation_dataset,  
    epochs=20  # Train for 20 epochs  
)  

plt.figure(figsize=(12, 4))  
plt.subplot(1, 2, 1)  
plt.plot(history.history['accuracy'], label='Train Accuracy')  
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  
plt.title('Model Accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')  
plt.legend()  

plt.subplot(1, 2, 2)  
plt.plot(history.history['loss'], label='Train Loss')  
plt.plot(history.history['val_loss'], label='Validation Loss')  
plt.title('Model Loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend()  

plt.show()  

model.save('weather_identification_model.h5')  

def predict_weather(image_path):  
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))  
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  
    img_array = tf.expand_dims(img_array, 0)  
    predictions = model.predict(img_array)  
    predicted_class = tf.argmax(predictions[0]).numpy()  
    class_names = train_dataset.class_names  
    return class_names[predicted_class]  

result = predict_weather("E:\\Weather Identification\\dataset\\cloud.jpg")  
result1 = predict_weather("E:\\Weather Identification\\dataset\\sun.jpg")  
print("Predicted Weather Condition:", result)  
print("Predicted Weather Condition:", result1)  

# Evaluate the model on the validation dataset  
val_loss, val_accuracy = model.evaluate(validation_dataset)  
print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")  

# Calculate accuracy for each weather condition  
class_names = train_dataset.class_names  
for i, class_name in enumerate(class_names):  
    class_accuracy = (validation_dataset.labels == i).mean()  
    print(f"Accuracy for {class_name}: {class_accuracy:.2f}")