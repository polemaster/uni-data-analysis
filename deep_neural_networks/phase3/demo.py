import os
import numpy as np
import tensorflow as tf
import string
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import wandb

from data_loader import load_and_process_train_data
from test_processing import process_custom_test_images

PROJECT_NAME = "sign-language-mnist-sweeps"
ENTITY = None

TEST_DIR = "ColabTestData"
OUTPUT_CSV_DIR = "processed_test_records" 
IMAGE_TO_DEMO = "ColabTestData/kuba1a.jpeg"

# Helper function
def get_letter_map():
    letters = string.ascii_uppercase
    mapping = {}
    idx = 0
    for char in letters:
        if char not in ['J', 'Z']:
            mapping[idx] = char
            idx += 1
    return mapping

letter_map = get_letter_map()
inv_letter_map = {v: k for k, v in letter_map.items()}

# Download & select the best model from WandB
def get_best_model_artifact(project, metric="test_accuracy"):
    """
    Scans a WandB project, finds the best run by selected metric,
    downloads its model artifact, and returns the model path.
    """
    print(f"Scanning WandB project '{project}' for best model...")
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}" if ENTITY else project)

    if not runs:
        raise ValueError("No runs found in this project.")

    # Sort runs by metric (descending)
    best_run = sorted(
        runs,
        key=lambda run: run.summary.get(metric, 0),
        reverse=True
    )[0]

    print(f"Best run found: {best_run.name}  (ID: {best_run.id})")
    print(f"Best {metric}: {best_run.summary.get(metric)}")

    artifacts = best_run.logged_artifacts()
    model_artifact = next((a for a in artifacts if a.type == "model"), None)

    if not model_artifact:
        raise ValueError("Best run does not contain any model artifact!")

    print(f"Downloading model artifact: {model_artifact.name}")
    artifact_dir = model_artifact.download()

    possible_extensions = [".keras", ".h5", ".hdf5"]

    for root, _, files in os.walk(artifact_dir):
        for f in files:
            if any(f.endswith(ext) for ext in possible_extensions):
                model_path = os.path.join(root, f)
                print(f"Detected model file: {model_path}")
                return model_path

    for root, dirs, files in os.walk(artifact_dir):
        if "saved_model.pb" in files:
            print(f"Detected SavedModel directory: {root}")
            return root

    raise ValueError(
        f"No recognized model file found in artifact: {artifact_dir}\n"
        "Expected .keras, .h5, or SavedModel format."
    )

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    wandb.login()

    # Load model
    print("\nFetching Best Model From WandB")
    model_path = get_best_model_artifact(PROJECT_NAME)
    print(f"Using model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Initialize Label Binarizer
    _, _, _, _, lb, _ = load_and_process_train_data()

    # Load test data
    print(f"\n Processing Test Images from {TEST_DIR} ")
    x_test, y_test_bin = process_custom_test_images(TEST_DIR, OUTPUT_CSV_DIR, lb)

    if len(x_test) == 0:
        print("No images processed. Exiting.")
        return

    # Normalize test data
    x_test = x_test.astype('float32') / 255.0
    
    # Convert one-hot encoded labels back to integers for metrics
    y_true = np.argmax(y_test_bin, axis=1)

    print(f"Loaded {len(x_test)} images.")

    # Run predictions
    print("\n Running Predictions ")
    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)

    # Report
    print("\n Classification Report ")
    class_labels = [letter_map[i] for i in range(len(letter_map))]
    
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion matrix
    print("\n Confusion Matrix ")
    plot_confusion_matrix(y_true, y_pred, class_labels)

    # Single Image Demo
    print("\n Single Image Prediction Demo ")
    if os.path.exists(IMAGE_TO_DEMO):
        img = Image.open(IMAGE_TO_DEMO).convert("L")
        img = img.resize((28, 28))
        
        x_demo = np.array(img).reshape(1, 28, 28, 1) / 255.0
        
        demo_pred = model.predict(x_demo)
        pred_id = np.argmax(demo_pred)
        pred_letter = letter_map[pred_id]

        print(f"Image: {IMAGE_TO_DEMO}")
        print(f"Predicted Letter: {pred_letter}")
        print("\nConfidence scores:")
        for i, score in enumerate(demo_pred[0]):
            print(f"{i:2d} ({letter_map[i]}): {score:.4f}")
    else:
        print(f"Demo image {IMAGE_TO_DEMO} not found.")

if __name__ == "__main__":
    main()