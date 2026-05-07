# =============================================================================
#  HANDWRITTEN DIGIT RECOGNITION - CNN MODEL TRAINING SCRIPT
#  Author: Rohit Yadav
# =============================================================================
#
#  This script trains a Convolutional Neural Network (CNN) on the MNIST dataset.
#  It is written step-by-step so that every single line can be explained during
#  a university viva or presentation.
#
#  WHAT THIS SCRIPT DOES:
#    1. Loads and preprocesses the MNIST dataset (60,000 training images)
#    2. Builds the CNN architecture layer by layer
#    3. Trains the model for up to 20 Epochs using the Adam Optimizer
#    4. Demonstrates Overfitting and Underfitting analysis with graphs
#    5. Evaluates on 10,000 unseen test images
#    6. Saves the final trained model as 'model_cnn.keras'
#
#  EXPECTED RESULT:
#    - Training Accuracy   : ~99.5%+
#    - Validation Accuracy : ~99.3%+  (on 6,000 held-out validation images)
#    - Test Accuracy       : ~99.36%  (on 10,000 completely unseen test images)
#
#  TO RUN THIS SCRIPT:
#    pip install tensorflow matplotlib scikit-learn
#    python train_model.py
# =============================================================================


# --- STEP 0: Import all required libraries ---
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and Keras: our deep learning framework
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Scikit-learn: for generating the advanced performance report
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =============================================================================
# STEP 1: LOAD THE MNIST DATASET
# =============================================================================
#
# MNIST is a collection of 70,000 grayscale images of handwritten digits (0-9).
# Each image is exactly 28 pixels wide and 28 pixels tall.
#
# Keras automatically splits it into:
#   x_train (60,000 images) -> used to TEACH the model
#   x_test  (10,000 images) -> used to TEST the model on images it has NEVER seen
#
# y_train and y_test are the CORRECT ANSWERS (labels) — integers from 0 to 9.
# =============================================================================

print("=" * 60)
print("STEP 1: Loading MNIST Dataset...")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"  Training images shape : {x_train.shape}")  # Should be (60000, 28, 28)
print(f"  Training labels shape : {y_train.shape}")  # Should be (60000,)
print(f"  Test images shape     : {x_test.shape}")   # Should be (10000, 28, 28)
print(f"  Test labels shape     : {y_test.shape}")   # Should be (10000,)
print(f"  Pixel value range     : {x_train.min()} to {x_train.max()}")


# =============================================================================
# STEP 2: PREPROCESS THE DATA
# =============================================================================
#
# PROBLEM: Raw pixel values are integers between 0 and 255.
# Large numbers make training unstable and slow because the gradients (the
# math calculations that update weights) become enormous.
#
# SOLUTION (Normalization):
#   Divide every pixel value by 255.0
#   This squishes all values into the range [0.0 to 1.0]
#   This helps the Adam optimizer converge much faster.
#
# RESHAPING:
#   The raw images are shape (28, 28) — a 2D grid.
#   Keras Conv2D layers expect a 3D input: (height, width, channels).
#   Since our images are grayscale (not RGB), they have 1 channel.
#   So we reshape to (28, 28, 1) using np.expand_dims.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Preprocessing Data...")
print("=" * 60)

# Normalize pixel values: 0-255 becomes 0.0-1.0
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# Add the channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test,  axis=-1)

print(f"  After reshape - Training : {x_train.shape}")   # (60000, 28, 28, 1)
print(f"  After reshape - Test     : {x_test.shape}")    # (10000, 28, 28, 1)
print(f"  After normalize - range  : {x_train.min():.1f} to {x_train.max():.1f}")


# =============================================================================
# STEP 3: VISUALISE A FEW SAMPLE IMAGES
# =============================================================================
#
# Before training, it is always good practice to visually inspect your data.
# This plot shows 10 random training images and their correct labels.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Saving sample images plot...")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample MNIST Training Images", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].squeeze(), cmap='gray')  # .squeeze() removes the channel dim for display
    ax.set_title(f"Label: {y_train[i]}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("output_sample_images.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: output_sample_images.png")


# =============================================================================
# STEP 4: BUILD THE CNN ARCHITECTURE
# =============================================================================
#
# A Convolutional Neural Network (CNN) is made of different types of layers.
# Each layer has a specific mathematical job.
#
# LAYER EXPLANATIONS:
#
#   Conv2D(32, (3,3), activation='relu')
#     - Applies 32 small 3x3 filters (kernels) that slide across the image.
#     - Each filter detects a different feature: edges, curves, corners.
#     - 'relu' activation: f(x) = max(0, x) — removes negative values.
#       This prevents the "Vanishing Gradient" problem that kills Sigmoid.
#
#   MaxPooling2D((2,2))
#     - Takes the maximum value from each 2x2 block.
#     - Shrinks the image by 50% to remove noise and reduce computation.
#     - Makes the model more robust to slight shifts in the digit position.
#
#   Dropout(0.25)
#     - Randomly turns off 25% of neurons during EACH training step.
#     - This forces the network NOT to rely on any single neuron.
#     - This is the PRIMARY tool used to PREVENT OVERFITTING.
#
#   Flatten()
#     - Converts the 2D feature maps into a 1D array.
#     - Example: (7, 7, 64) feature map -> 3136 values in a single row.
#
#   Dense(128, activation='relu')
#     - A fully connected layer. Every neuron connects to every input.
#     - This layer learns high-level combinations of all detected features.
#
#   Dense(10, activation='softmax')
#     - The OUTPUT layer. 10 neurons = 10 possible digits (0-9).
#     - Softmax converts raw scores into probabilities that sum to 1.0.
#     - The neuron with the HIGHEST probability is the predicted digit.
#
# TOTAL PARAMETERS: ~475,434 learnable weights
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Building CNN Architecture...")
print("=" * 60)

model = Sequential([

    # --- BLOCK 1: Learn basic features (edges, simple curves) ---
    Conv2D(32, (3, 3), padding='same', activation='relu',
           input_shape=(28, 28, 1), name='conv1'),

    Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2'),

    MaxPooling2D((2, 2), name='pool1'),

    Dropout(0.25, name='dropout1'),


    # --- BLOCK 2: Learn complex features (loops, intersections) ---
    Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3'),

    Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4'),

    MaxPooling2D((2, 2), name='pool2'),

    Dropout(0.25, name='dropout2'),


    # --- BLOCK 3: Classification Head ---
    Flatten(),

    Dense(128, activation='relu', name='dense1'),
    Dropout(0.5, name='dropout3'),

    Dense(64, activation='relu', name='dense2'),
    Dropout(0.5, name='dropout4'),

    # Output: 10 neurons, one per digit (0-9)
    Dense(10, activation='softmax', name='output_layer')
])

# Print the full summary (shows layer shapes and parameter counts)
model.summary()


# =============================================================================
# STEP 5: COMPILE THE MODEL
# =============================================================================
#
# Compiling defines:
#   1. OPTIMIZER (Adam): The algorithm that updates the weights during training.
#      - Adam adapts the learning rate per-parameter using momentum (m_t) and
#        RMSprop (v_t). This makes it faster and more stable than plain SGD.
#      - learning_rate=0.001 is the default and generally the best starting value.
#
#   2. LOSS FUNCTION (sparse_categorical_crossentropy):
#      - For multi-class classification with integer labels (0, 1, 2...9).
#      - Cross-Entropy penalizes CONFIDENT WRONG predictions extremely heavily.
#        Formula: Loss = -log(P(correct class))
#        If the model is 99% sure of the WRONG digit, the loss is enormous.
#        This forces aggressive weight updates, making the model learn fast.
#
#   3. METRICS (accuracy):
#      - The percentage of correct predictions. Displayed during each epoch.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Compiling the Model...")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("  Optimizer  : Adam (learning_rate=0.001)")
print("  Loss       : sparse_categorical_crossentropy")
print("  Metrics    : accuracy")


# =============================================================================
# STEP 6: DEFINE TRAINING CALLBACKS
# =============================================================================
#
# Callbacks are functions that automatically run at the end of each Epoch.
# They give the training process "intelligence" beyond just running blindly.
#
#   EarlyStopping:
#     - Monitors the VALIDATION LOSS after every epoch.
#     - If it stops improving for 4 consecutive epochs, STOP training.
#     - restore_best_weights=True: rewinds back to the BEST epoch automatically.
#     - This prevents OVERFITTING (the model memorizing training data).
#
#   ReduceLROnPlateau:
#     - If validation loss doesn't improve for 2 epochs, the learning rate
#       is multiplied by 0.5 (halved).
#     - This allows fine-grained tuning when the model gets close to perfect.
#
#   ModelCheckpoint:
#     - Automatically saves the model to 'model_cnn.keras' after each epoch
#       IF the validation loss improved.
#     - Guarantees we always have the BEST version saved.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Setting Up Callbacks...")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'model_cnn.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("  EarlyStopping      : patience=4 epochs on val_loss")
print("  ReduceLROnPlateau  : halves learning rate if val_loss stalls for 2 epochs")
print("  ModelCheckpoint    : saves best model to model_cnn.keras")


# =============================================================================
# STEP 7: TRAIN THE MODEL (THE MOST IMPORTANT STEP)
# =============================================================================
#
# model.fit() is the training loop. This is where the math happens.
#
# WHAT HAPPENS INSIDE EACH EPOCH:
#   1. All 60,000 training images are fed to the model in batches of 128.
#   2. For each batch, the model makes predictions (forward pass).
#   3. The Cross-Entropy loss is calculated against the correct labels.
#   4. Backpropagation runs the Chain Rule through all layers to calculate
#      how much EACH WEIGHT contributed to the error.
#   5. Adam optimizer updates all ~475K weights using the momentum formulas.
#   6. At the end of all batches, the model is tested on the VALIDATION SET.
#      (10% of training data = 6,000 images held aside)
#
# PARAMETERS:
#   epochs=20           : Run the full training loop up to 20 times.
#   batch_size=128      : Process 128 images at once before updating weights.
#   validation_split=0.1: Reserve 10% of training data for validation.
#
# EPOCH ACCURACY PROGRESSION (expected):
#   Epoch 1   : ~85-90% (model is learning fast from scratch)
#   Epoch 5   : ~98%    (basic digit shapes fully learned)
#   Epoch 10+ : ~99%+   (fine-tuning subtle differences like '4' vs '9')
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Training the Model (up to 20 Epochs)...")
print("=" * 60)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\n  Training complete!")


# =============================================================================
# STEP 8: OVERFITTING AND UNDERFITTING ANALYSIS
# =============================================================================
#
# UNDERFITTING: When the model is too simple and cannot learn the training data.
#   - Sign: Both training AND validation accuracy are LOW.
#   - This usually happens in early epochs (Epoch 1-3).
#
# OVERFITTING: When the model memorizes the training data but fails on new data.
#   - Sign: Training accuracy is HIGH but validation accuracy is LOWER or falling.
#   - In our model, Dropout layers are the main protection against this.
#
# THE GRAPHS BELOW SHOW:
#   - Accuracy Graph: Training accuracy vs Validation accuracy per epoch.
#     If the lines stay CLOSE TOGETHER, the model is NOT overfitting.
#
#   - Loss Graph: Training loss vs Validation loss per epoch.
#     If validation loss starts GOING UP while training loss goes DOWN,
#     the model is beginning to overfit (Dropout and EarlyStopping prevent this).
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Generating Overfitting / Underfitting Analysis Graphs...")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Progress — Overfitting & Underfitting Analysis",
             fontsize=14, fontweight='bold')

epochs_ran = range(1, len(history.history['accuracy']) + 1)

# --- Plot 1: Accuracy ---
ax1.plot(epochs_ran, history.history['accuracy'],     'b-o', label='Training Accuracy',   linewidth=2)
ax1.plot(epochs_ran, history.history['val_accuracy'], 'r-o', label='Validation Accuracy', linewidth=2)
ax1.set_title("Accuracy per Epoch", fontsize=12)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.8, 1.01])

# Annotate the best validation accuracy point
best_val_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
best_val_acc   = max(history.history['val_accuracy'])
ax1.annotate(f"Best: {best_val_acc:.4f}\n(Epoch {best_val_epoch})",
             xy=(best_val_epoch, best_val_acc),
             xytext=(best_val_epoch + 0.5, best_val_acc - 0.02),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=9, color='green')

# --- Plot 2: Loss ---
ax2.plot(epochs_ran, history.history['loss'],     'b-o', label='Training Loss',   linewidth=2)
ax2.plot(epochs_ran, history.history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
ax2.set_title("Loss per Epoch", fontsize=12)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add a shaded "Underfitting Zone" for early epochs
if len(epochs_ran) > 2:
    ax2.axvspan(1, min(3, len(epochs_ran)), alpha=0.1, color='orange',
                label='Underfitting Zone (Early Epochs)')
    ax2.legend()

plt.tight_layout()
plt.savefig("output_training_curves.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: output_training_curves.png")
print("  ANALYSIS:")
print("    - If Training Acc >> Validation Acc → Overfitting (Dropout prevents this)")
print("    - If both are LOW → Underfitting (more epochs or bigger model needed)")
print("    - If both are HIGH and CLOSE → Perfect generalization ✓")


# =============================================================================
# STEP 9: EVALUATE ON THE TEST SET (FINAL SCORE)
# =============================================================================
#
# The model has NEVER seen these 10,000 images.
# This is the true, unbiased measurement of real-world accuracy.
#
# 'verbose=0' means we suppress the progress bar here.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Evaluating on 10,000 Unseen Test Images...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\n  Final Test Loss     : {test_loss:.4f}")
print(f"  Final Test Accuracy : {test_accuracy * 100:.2f}%")


# =============================================================================
# STEP 10: PER-CLASS PERFORMANCE REPORT (Precision, Recall, F1-Score)
# =============================================================================
#
# Overall accuracy tells us "what % was correct", but it hides per-digit details.
# The Classification Report breaks performance down PER DIGIT:
#
#   Precision: Out of all predictions of digit X, how many were actually X?
#   Recall   : Out of all ACTUAL digit X images, how many did we correctly find?
#   F1-Score : The harmonic mean of Precision and Recall. Best single metric.
#
# The Confusion Matrix shows a grid of:
#   Row = ACTUAL digit  /  Column = PREDICTED digit
#   Diagonal cells = CORRECT predictions (we want all mass on the diagonal)
#   Off-diagonal cells = MISTAKES (e.g., predicted '4' when actual was '9')
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Per-Digit Performance Report...")
print("=" * 60)

y_pred_probs = model.predict(x_test, verbose=0)       # Get raw probabilities
y_pred       = np.argmax(y_pred_probs, axis=1)        # Get the predicted digit

print("\n  Classification Report (per digit 0-9):")
print("  " + "-" * 55)
print(classification_report(y_test, y_pred,
                             target_names=[f"Digit {i}" for i in range(10)]))

# --- Plot Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[str(i) for i in range(10)])
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title(f"Confusion Matrix (Test Accuracy: {test_accuracy*100:.2f}%)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("output_confusion_matrix.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: output_confusion_matrix.png")


# =============================================================================
# STEP 11: SAVE THE FINAL MODEL
# =============================================================================
#
# The model is already saved by ModelCheckpoint during training, but we also
# call model.save() here explicitly to guarantee the final state is persisted.
#
# The .keras format is TensorFlow's modern native format. It stores:
#   - The full architecture (all layers and their configuration)
#   - The trained weight values (~475K floats)
#   - The compile configuration (optimizer, loss, metrics)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Saving Model to model_cnn.keras...")
print("=" * 60)

model.save('model_cnn.keras')
print("  Model saved successfully as 'model_cnn.keras'")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Total Epochs Run       : {len(history.history['accuracy'])}")
print(f"  Final Test Accuracy    : {test_accuracy * 100:.2f}%")
print(f"  Final Test Loss        : {test_loss:.4f}")
print(f"  Model saved to         : model_cnn.keras")
print(f"\n  Output files generated:")
print(f"    - output_sample_images.png   (10 sample training images)")
print(f"    - output_training_curves.png (Accuracy & Loss graphs per epoch)")
print(f"    - output_confusion_matrix.png (Per-digit prediction heatmap)")
print("=" * 60)
