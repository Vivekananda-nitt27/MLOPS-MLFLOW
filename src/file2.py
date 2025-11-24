import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# One-hot encoding labels
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameters
epochs = 50
batch_size = 8
learning_rate = 0.001

# Set Experiment Name
mlflow.set_experiment("YT-MLOPS-Exp3")

with mlflow.start_run() as run:
    # Define ANN Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=0)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Log parameters & metrics
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)

    # Confusion Matrix Logging
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=wine.target_names, yticklabels=wine.target_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix_ann.png")
    mlflow.log_artifact("confusion_matrix_ann.png")

    # Save ANN model in MLflow
    mlflow.tensorflow.log_model(model, "ANN-Wine-Model")

    # Tags
    mlflow.set_tags({"Author": "Vivek", "Model": "ANN", "Project": "Wine Classification"})

    print("Accuracy:", accuracy)
    print("Experiment logged successfully ✔")
