import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import calculate_all_metrics, print_metrics

# Enable memory growth for GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used for training.")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found. Training will use CPU.")

# Set parameters
data_dir = 'pest'
img_height, img_width = 224, 224  # MobileNetV2 expects 224x224
batch_size = 32  # Increased batch size
epochs = 20  # Changed to 20 epochs

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Reduced rotation
    width_shift_range=0.15,  # Reduced shift
    height_shift_range=0.15,
    shear_range=0.1,  # Reduced shear
    zoom_range=0.1,  # Reduced zoom
    horizontal_flip=True,
    fill_mode='nearest',  # Changed to nearest for faster processing
    brightness_range=[0.9, 1.1]  # Reduced brightness range
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_gen.num_classes

# Create MobileNetV2 model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Reduced dense layer size
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Reduced dropout
    layers.Dense(num_classes, activation='softmax')
])

# Use mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # Reduced patience for faster convergence
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=2,  # Reduced patience for faster learning rate adjustment
    min_lr=1e-6,
    verbose=1
)

# Calculate class weights
class_indices = train_gen.class_indices
labels = []
for class_name, idx in class_indices.items():
    class_path = os.path.join(data_dir, class_name)
    n_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    labels.extend([idx] * n_images)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# Train the model
print("\nStarting training...")
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Save the model
model.save('mobilenetv2_model.keras')
print("\nModel saved as 'mobilenetv2_model.keras'")

# Save training history
import json
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}
with open('mobilenetv2_history.json', 'w') as f:
    json.dump(history_dict, f)

# Print final metrics
val_gen.reset()
y_pred = model.predict(val_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(class_indices.keys())))

# Calculate and print detailed metrics
metrics = calculate_all_metrics(y_true, y_pred_classes)
print_metrics(metrics, list(class_indices.keys()))

# Save metrics to file
metrics_dict = {class_name: metrics[f'class_{idx}'] for class_name, idx in class_indices.items()}
with open('mobilenetv2_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4) 