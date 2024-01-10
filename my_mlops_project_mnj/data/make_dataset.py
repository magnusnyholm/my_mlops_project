import os
import torch

if __name__ == '__main__':

    # Current script directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the project root directory
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

    # Paths to the 'data/processed' and 'data/raw' directories
    data_processed_dir = os.path.join(project_root_dir, 'data', 'processed')
    data_raw_dir = os.path.join(project_root_dir, 'data', 'raw')

    train_data, train_labels = [], []

    for i in range(5):
        train_data.append(torch.load(os.path.join(data_raw_dir, f"train_images_{i}.pt")))
        train_labels.append(torch.load(os.path.join(data_raw_dir, f"train_target_{i}.pt")))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Save the processed tensors
    torch.save(train_data, os.path.join(data_processed_dir, 'processed_images_train.pt'))
    torch.save(train_labels, os.path.join(data_processed_dir, 'processed_target_train.pt'))
