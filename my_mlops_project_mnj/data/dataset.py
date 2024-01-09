import torch

def mnist():

    """Return train and test dataloaders for MNIST."""

    train_data = torch.load("../data/processed/processed_images_train.pt")
    train_labels = torch.load("../data/processed/processed_target_train.pt")
    test_data = torch.load("../data/raw/test_images.pt")
    test_labels = torch.load("../data/raw/test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (torch.utils.data.TensorDataset(train_data, train_labels), torch.utils.data.TensorDataset(test_data, test_labels))