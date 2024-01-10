import click
import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path

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

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=5, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    model = myawesomemodel.to(device)
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_values = [] #used for accuracy plot

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")
        loss_values.append(loss.item())
    
    # Saving model
    save_directory = project_root_dir / 'models'
    save_directory.mkdir(exist_ok=True)
    saving_path = save_directory / "trained_model.pt"
    torch.save(model, saving_path)
    print(f"Model was saved to {saving_path}")

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
    print(f"Plot of training curve was saved to {s_path}")

cli.add_command(train)

if __name__ == "__main__":
    cli()