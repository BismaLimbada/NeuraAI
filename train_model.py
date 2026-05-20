import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from train_data import X_train, y_train, all_words, tags
import json

# 1. Hyperparameters Configuration
INPUT_SIZE = len(all_words)
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 8

print("--- Initializing Neural Network Architecture ---")
print(f"Input features (Vocabulary size): {INPUT_SIZE}")
# Convert target labels to one-hot encoded vectors for categorical cross-entropy loss
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)

# 2. Build the Feed-Forward Neural Network Model
model = Sequential([
    # Input layer + First hidden layer with Dropout to prevent overfitting
    Dense(HIDDEN_SIZE_1, input_shape=(INPUT_SIZE,), activation='relu'),
    Dropout(0.3), 
    
    # Second deep hidden layer
    Dense(HIDDEN_SIZE_2, activation='relu'),
    Dropout(0.3),
    
    # Output layer using Softmax to produce probability scores for each tag
    Dense(OUTPUT_SIZE, activation='softmax')
])

# 3. Compile the Model with Adam Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train the Deep Learning Model
print("\n--- Training Model ---")
history = model.fit(
    X_train, 
    y_train_encoded, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1
)

# 5. Save the trained model and parameters for production deployment
model.save('mental_health_model.keras')

# Save tracking configurations (vocabulary and tags) so our backend can map inputs properly
data_mappings = {
    "all_words": all_words,
    "tags": tags
}

with open("data_mappings.json", "w") as f:
    json.dump(data_mappings, f)

print("\nPhase 2 Complete! Model trained successfully and saved as 'mental_health_model.keras'.")