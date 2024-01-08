import os
import torch


if __name__ == '__main__':

    # Define paths
    raw_data_folder = 'data/raw'
    processed_data_folder = 'data/processed'

    # Initialize lists to store tensors
    image_tensors = []
    target_tensors = []

    # Step 1: Loop through the raw data folder and separate tensors
    for filename in os.listdir(raw_data_folder):
        filepath = os.path.join(raw_data_folder, filename)
        
        if "train_images" in filename and filename.endswith('.pt'):
            image_tensor = torch.load(filepath)
            image_tensors.append(image_tensor)
        
        elif "train_target" in filename and filename.endswith('.pt'):
            target_tensor = torch.load(filepath)
            target_tensors.append(target_tensor)

    # Convert the lists of tensors into single tensors
    combined_image_tensor = torch.stack(image_tensors) if image_tensors else None
    combined_target_tensor = torch.stack(target_tensors) if target_tensors else None

    # Normalize and save the image tensor if it exists
    if combined_image_tensor is not None:
        combined_image_tensor = combined_image_tensor.float()
        mean = combined_image_tensor.mean()
        std = combined_image_tensor.std()
        normalized_image_tensor = (combined_image_tensor - mean) / std
        output_image_filepath = os.path.join(processed_data_folder, 'processed_images_train.pt')
        torch.save(normalized_image_tensor, output_image_filepath)
        print(f"Normalized and processed image tensor saved to {output_image_filepath}")

    # Normalize and save the target tensor if it exists
    if combined_target_tensor is not None:
        combined_target_tensor = combined_target_tensor.float()
        mean_target = combined_target_tensor.mean()
        std_target = combined_target_tensor.std()
        normalized_target_tensor = (combined_target_tensor - mean_target) / std_target
        output_target_filepath = os.path.join(processed_data_folder, 'processed_target_train.pt')
        torch.save(normalized_target_tensor, output_target_filepath)
        print(f"Normalized and processed target tensor saved to {output_target_filepath}")

