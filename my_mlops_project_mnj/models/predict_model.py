import sys
import os
# Add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import click
import torch
from data.dataset import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the 'models' directory
models_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', 'models'))

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.argument("model_filename")
def evaluate(model_filename):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    model_checkpoint = os.path.join(models_dir, model_filename)
    print(model_checkpoint)

    # Rest of the code remains the same
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print(f'Evaluation accuracy: {round((test_preds == test_labels).float().mean().item(),2)}')

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()