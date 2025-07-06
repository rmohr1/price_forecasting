from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from price_forecasting.utils.scoring_tools import get_mean_crps


def quantile_loss(
        preds: Sequence,
        target: Sequence,
        quantiles: Sequence,
    ):
    """Calculate quantile loss function.

    Args:
        preds: quantile predictions from model
        target: target value to be trained on
        quantles: list of quantile levels

    Returns:
        loss: 
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

def train(
        model: torch.nn, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        y_scaler: StandardScaler,
        config: dict, 
        SAVE_PATH: Path,
        device: torch.device,
    ):
    """Train a torch model. Saves result to config["save_path"].

    Args:
        model: torch model to be trained
        train_loader: training set DataLoader object
        test_loader: test set DataLoader object
        y_scaler: scaler for y data
        config: dict of config values
        save_path: directory to save model
        device: torch device type (cpu, gpu)

    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    best_crps = float('inf')
    y_test = [y for x, y in test_loader]
    y_test = torch.cat(y_test)
    y_test = y_scaler.inverse_transform(y_test)
    y_test = y_test.reshape([-1])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = quantile_loss(preds, y, model.quantiles)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #test_loss = evaluate(model, test_loader, device)
        #print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, \
        #      Test Loss {test_loss:.4f}")
        #if test_loss < best_val_loss:
        #    best_val_loss = test_loss
        #    torch.save(model.state_dict(), SAVE_PATH / 'model_wts_loss.pt')

        y_pred = predict(model, test_loader, device)
        y_pred = y_scaler.inverse_transform(y_pred)
        crps = get_mean_crps(y_pred, y_test, model.quantiles)
        print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, \
              CRPS {crps:.4f}")
        if crps < best_crps:
            best_crps = crps
            torch.save(model.state_dict(), SAVE_PATH / 'model_wts.pt')
    
    model.load_state_dict(torch.load(SAVE_PATH / 'model_wts.pt'))
    y_pred = predict(model, test_loader, device)
    return y_pred
    

def evaluate(model, test_loader, device) -> float:
    """Evaluate a torch model against test set.

    Args:
        model: torch model to be trained
        test_loader: test set DataLoader object
        device: torch device type (cpu, gpu)
    
    Returns:
        mean quantile loss over test set

    """
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = quantile_loss(preds, y, model.quantiles)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def predict(model, test_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            yi = model(x)
            y_pred.append(yi)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.reshape([-1, y_pred.shape[-1]])
    return y_pred