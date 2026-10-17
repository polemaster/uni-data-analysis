import os
import string
import numpy as np
import pandas as pd
from PIL import Image

def process_custom_test_images(input_folder, output_folder, lb_instance):
    """
    Reads images from input_folder, converts to grayscale 28x28, 
    and returns processed numpy arrays.
    
    Args:
        input_folder: Path to raw images.
        output_folder: Path to save the processed CSV (for records).
        lb_instance: The LabelBinarizer fitted on the training data.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Mapping logic
    all_letters = string.ascii_uppercase
    letter_to_label = {}
    for i, char in enumerate(all_letters):
        if char not in ['J', 'Z']:
            letter_to_label[char] = i

    csv_data = []
    
    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print(f"Warning: Test Data Input folder {input_folder} is empty or missing.")
        return np.array([]), np.array([])

    print(f"Processing custom images from: {input_folder}")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert('L') # grayscale
                img = img.resize((28, 28)) 
                
                # Derive label from filename (e.g., "A.jpg" -> "A")
                label_char = filename.split('.')[0][-1].upper()
                
                if label_char in letter_to_label:
                    label = letter_to_label[label_char]
                    img_array = np.array(img).flatten()
                    row_data = [label] + img_array.tolist()
                    csv_data.append(row_data)
                else:
                    print(f"Skipping {filename}: Label '{label_char}' invalid.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save to CSV for reference
    headers = ['label'] + [f'pixel{i+1}' for i in range(28*28)]
    output_csv = os.path.join(output_folder, "test_images.csv")
    
    if csv_data:
        df = pd.DataFrame(csv_data, columns=headers)
        df.to_csv(output_csv, index=False)
        
        # Prepare for Model
        y_test = df['label']
        x_test = df.drop('label', axis=1)
        x_test = x_test.values.reshape(-1, 28, 28, 1)
        
        # Transform labels using the Training Logic (lb_instance)
        y_test_bin = lb_instance.transform(y_test)
        
        print(f"Custom Test Data Processed: {x_test.shape}")
        return x_test, y_test_bin
    else:
        print("No valid images found in custom folder.")
        return np.array([]), np.array([])