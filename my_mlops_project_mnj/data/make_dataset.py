import torch

if __name__ == '__main__':

    # Define saving path
    save_directory = '../data/processed/'

    train_data, train_labels = [ ], [ ]

    for i in range(5):
        train_data.append(torch.load(f"../data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"../data/raw/train_target_{i}.pt"))
    
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Save the normalized tensors
    torch.save(train_data, save_directory + 'processed_images_train.pt')
    torch.save(train_labels, save_directory + 'processed_target_train.pt')