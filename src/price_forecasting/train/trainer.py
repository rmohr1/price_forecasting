import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def train(
        model: torch.nn, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        y_scaler: StandardScaler,
        config: dict, 
        SAVE_PATH: Path,
        device: torch.device,
        epoch_grade: str="crps",
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
        epoch_grade: function to decide and save the best epoch. crps or loss
    """
    model.to(device)
    if "weight_decay" in config:
        wd = config['weight_decay']
    else:
        wd = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=wd)

    best_val_loss = float('inf')
    #best_crps = float('inf')
    #y_test = [y for x, y in test_loader]
    #y_test = torch.cat(y_test)
    #y_test = y_scaler.inverse_transform(y_test)
    #y_test = y_test.reshape([-1])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = model.loss(preds, y)
            loss.mean().backward()
            optimizer.step()
            total_loss += loss.sum()

        test_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader.dataset)/288:.4f}, \
              Test Loss {test_loss/len(test_loader.dataset)/288:.4f}")
        #print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader.dataset):.4f}, \
        #      Test Loss {test_loss:.4f}")
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            save_model(model, test_loader, device, y_scaler, SAVE_PATH)

    return best_val_loss

    """
        if epoch_grade == "loss":
            test_loss = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, \
                  Test Loss {test_loss:.4f}")
            if test_loss < best_val_loss and total_loss/len(train_loader) < 1.0:
                best_val_loss = test_loss
                torch.save(model.state_dict(), SAVE_PATH / 'model_wts.pt')
                y_pred = predict(model, test_loader, device)
                y_pred = y_scaler.inverse_transform(y_pred)
                np.save(SAVE_PATH / 'y_pred.npy', y_pred)

        elif epoch_grade == "crps":
            y_pred = predict(model, test_loader, device)
            y_pred = y_scaler.inverse_transform(y_pred)
            crps = get_mean_crps(y_pred, y_test, model.quantiles)
            print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, \
                  CRPS {crps:.4f}")
            if (crps < best_crps) and total_loss/len(train_loader) < 1.0:
                best_crps = crps
                torch.save(model.state_dict(), SAVE_PATH / 'model_wts.pt')
                y_pred = predict(model, test_loader, device)
                y_pred = y_scaler.inverse_transform(y_pred)
                np.save(SAVE_PATH / 'y_pred.npy', y_pred)
        else:
            raise ValueError("epoch_grade not recognized. Must be loss or crps")
    if epoch_grade == "loss":
        return best_val_loss
    elif epoch_grade == "crps":
        return best_crps
    """
    

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
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = model.loss(preds, y)
            total_loss += loss.sum()
    return total_loss

def save_model(model, test_loader, device, y_scaler, SAVE_PATH):
    model.eval()
    y_pred = None
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            if y_pred is None:
                y_pred = preds
            else:
                y_pred = {k: torch.cat([y_pred[k], preds[k]], dim=0) for k in y_pred}
    
    y_pred = {k: y_pred[k].numpy() for k in y_pred}
    torch.save(model.state_dict(), SAVE_PATH / 'model_wts.pt')
    np.savez(SAVE_PATH / 'y_pred.npz', **y_pred)
    with open(SAVE_PATH / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)