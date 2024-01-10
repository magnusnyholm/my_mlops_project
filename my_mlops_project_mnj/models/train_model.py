import click
import os
import sys
import torch
import hydra
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import logging
log = logging.getLogger(__name__)


# Add the parent directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root_dir = Path(os.path.dirname(os.path.dirname(script_dir)))  # Navigate two levels up
sys.path.append(parent_dir)

# Import model and dataset
from models.model import myawesomemodel
from data.dataset import mnist

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="../../config/", config_name="train_config.yaml", version_base="1.3.2")
def main(cfg):
    # Print hyperparameters for verification
    log.info(f"Batch size: {cfg.hyperparameters.batch_size}, "
          f"Epochs: {cfg.hyperparameters.epochs}, "
          f"Learning rate: {cfg.hyperparameters.learning_rate}")

    # Call the train function with parameters from cfg
    train(cfg.hyperparameters.learning_rate, 
          cfg.hyperparameters.batch_size, 
          cfg.hyperparameters.epochs)

def train(learning_rate, batch_size, epochs):
    """Train a model on MNIST."""
    log.info("Training day and night")
    log.info(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")

    # TODO: Implement training loop here
    model = myawesomemodel.to(device)
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_values = [] #used for accuracy plot

    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch} Loss {loss}")
        loss_values.append(loss.item())
    
    # Saving model
    save_directory = project_root_dir / 'models'
    save_directory.mkdir(exist_ok=True)
    saving_path = save_directory / "trained_model.pt"
    torch.save(model, saving_path)
    log.info(f"Model was saved to {saving_path}")

    #Training figure
    #Plot the training curve
    plt.figure()
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_path = project_root_dir / 'reports' / 'figures'
    save_path.mkdir(exist_ok=True)
    s_path = save_path / 'training_loss_curve.png'
    plt.savefig(s_path)
    log.info(f"Plot of training curve was saved to {s_path}")

if __name__ == "__main__":
    main()