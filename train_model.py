import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from train_data import X_train, y_train, tags
import json

# 1. Hyperparameters Configuration
INPUT_SIZE = X_train.shape[1]  # 512-dim Universal Sentence Encoder embedding
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 8

print("--- Initializing Neural Network Architecture ---")
print(f"Input features (USE embedding size): {INPUT_SIZE}")
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)

# 2. Build the Feed-Forward Neural Network Model
# Embeddings are dense, continuous, and much richer per-input than
# bag-of-words, so with a small pattern set we lean a bit more on
# regularization (L2 + dropout) to avoid overfitting to the exact
# training sentences.
model = Sequential([
    Dense(HIDDEN_SIZE_1, input_shape=(INPUT_SIZE,), activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.4),

    Dense(HIDDEN_SIZE_2, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.4),

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

# Save tracking configuration (tags only - no vocabulary needed anymore
# since inference re-embeds the raw sentence with the same USE model)
data_mappings = {
    "tags": tags
}

with open("data_mappings.json", "w") as f:
    json.dump(data_mappings, f)

print("\nPhase 2 Complete! Model trained successfully and saved as 'mental_health_model.keras'.")
