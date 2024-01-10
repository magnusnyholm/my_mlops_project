import os
import torch

def mnist():
    """Return train and test dataloaders for MNIST."""

    # Current script directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the project root directory
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

    # Paths to the 'data/processed' and 'data/raw' directories
    data_processed_dir = os.path.join(project_root_dir, 'data', 'processed')
    data_raw_dir = os.path.join(project_root_dir, 'data', 'raw')

    # Loading the data using the absolute paths
    train_data = torch.load(os.path.join(data_processed_dir, "processed_images_train.pt"))
    train_labels = torch.load(os.path.join(data_processed_dir, "processed_target_train.pt"))
    test_data = torch.load(os.path.join(data_raw_dir, "test_images.pt"))
    test_labels = torch.load(os.path.join(data_raw_dir, "test_target.pt"))

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (torch.utils.data.TensorDataset(train_data, train_labels), torch.utils.data.TensorDataset(test_data, test_labels))

mnist()
