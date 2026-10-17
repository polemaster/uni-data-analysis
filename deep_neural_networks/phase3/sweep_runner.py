import os
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

from data_loader import load_and_process_train_data
from test_processing import process_custom_test_images
from model_builder import build_model

# SETUP PATHS 
RAW_TEST_IMAGES_DIR = "./ColabTestData"
PROCESSED_TEST_OUTPUT_DIR = "./test_images_processed"
SWEEP_PROJECT = "sign-language-mnist-sweeps"

# LOAD DATA 
x_train, y_train_bin, x_val, y_val_bin, lb, num_classes = load_and_process_train_data()
x_test, y_test_bin = process_custom_test_images(RAW_TEST_IMAGES_DIR, PROCESSED_TEST_OUTPUT_DIR, lb)

# DEFINE SWEEP CONFIGURATION
sweep_configuration = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"value": 15},
        "batch_size": {"values": [64, 128]},
        "learning_rate": {"min": 0.0001, "max": 0.005},
        "optimizer": {"values": ["adam", "rmsprop"]},
        "dropout": {"min": 0.2, "max": 0.5},
        "dense_units": {"values": [128, 256, 512]},
        # Data Augmentation Params
        "rotation_range": {"min": 0, "max": 20},
        "zoom_range": {"min": 0.0, "max": 0.2}
    }
}

# DEFINE TRAINING FUNCTION
def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Dynamic Data Augmentation based on Sweep Config
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.rotation_range,
            zoom_range=config.zoom_range,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Build Model
        model = build_model(config, num_classes)

        # Create a checkpoint callback
        model_filename = f"model-{wandb.run.id}.keras"
        checkpoint = ModelCheckpoint(
            filepath=model_filename,
            monitor='val_accuracy',
            save_best_only=True, 
            verbose=1
        )

        # Train
        history = model.fit(
            datagen.flow(x_train, y_train_bin, batch_size=config.batch_size),
            epochs=config.epochs,
            validation_data=val_datagen.flow(x_val, y_val_bin, batch_size=config.batch_size),
            callbacks=[
                ReduceLROnPlateau(patience=2, factor=0.5, min_lr=0.00001),
                WandbMetricsLogger(),
                checkpoint
            ]
        )


        # Evaluate on Custom Test Set
        if len(x_test) > 0:
            print("Evaluating on Custom Test Set")
            test_loss, test_acc = model.evaluate(val_datagen.flow(x_test, y_test_bin))
            wandb.summary['test_loss'] = test_loss
            wandb.summary['test_accuracy'] = test_acc

        
        # LOG THE ARTIFACT
        if os.path.exists(model_filename):
            final_val_acc = max(history.history['val_accuracy'])
            
            if final_val_acc > 0.80:
                print(f"Uploading model with accuracy {final_val_acc}")
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", 
                    type="model",
                    metadata={"val_accuracy": final_val_acc}
                )
                artifact.add_file(model_filename)
                wandb.log_artifact(artifact)
            else:
                print("Model accuracy too low. Not uploading artifact.")


# EXECUTE
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_configuration, project=SWEEP_PROJECT)
    print("Starting Sweep Agent (Running 5 experiments)")
    wandb.agent(sweep_id, train_sweep, count=5)
