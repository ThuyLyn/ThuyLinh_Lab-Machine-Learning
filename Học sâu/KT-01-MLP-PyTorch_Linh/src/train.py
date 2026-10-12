import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from dataset import ROOT, make_loaders, prepare_data
from model import MLP, count_params


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_criterion(y_train, device):
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def make_optimizer(name, params, lr=1e-3):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-2, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    raise ValueError(f"optimizer không hợp lệ: {name}")

def evaluate(model, loader, criterion, device, use_eval=True):
    model.eval() if use_eval else model.train()
    probs, ys, total = [], [], 0.0
    with torch.no_grad():                               
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += criterion(logits, yb).item() * len(yb)
            probs.append(torch.sigmoid(logits).cpu()); ys.append(yb.cpu())
    probs = np.nan_to_num(torch.cat(probs).numpy().ravel(), nan=0.5)  
    ys = torch.cat(ys).numpy().ravel()
    return total / len(ys), average_precision_score(ys, probs)

def train(model, optimizer, criterion, train_loader, val_loader, device, epochs=50,
          zero_grad=True, use_eval=True, patience=None, ckpt="models/best.pt", verbose=True):
    hist = {"train_loss": [], "val_loss": [], "val_prauc": []}
    best, wait = -1.0, 0
    if patience is not None:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()                                     
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if zero_grad:
                optimizer.zero_grad()                    
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        vl, vp = evaluate(model, val_loader, criterion, device, use_eval)
        hist["train_loss"].append(running / len(train_loader.dataset))
        hist["val_loss"].append(vl); hist["val_prauc"].append(vp)
        if verbose:
            print(f"epoch {epoch + 1:3d} | train {hist['train_loss'][-1]:.4f} | val {vl:.4f} | PR-AUC {vp:.4f}")
        if patience is not None:                           
            if vp > best:
                best, wait = vp, 0
                torch.save(model.state_dict(), ckpt)        
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping tại epoch {epoch + 1}")
                    break
    return hist

def plot_learning_curve(hist, path):
    best_ep = int(np.argmax(hist["val_prauc"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].axvline(best_ep, ls="--", c="gray", label="epoch tốt nhất")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["val_prauc"]); ax[1].axvline(best_ep, ls="--", c="gray")
    ax[1].set_title("Val PR-AUC"); ax[1].set_xlabel("epoch")
    plt.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="đường dẫn CSV (mặc định: data/WA_Fn-UseC_-Telco-Customer-Churn.csv)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-zero-grad", action="store_true", help="THÍ NGHIỆM 1: bỏ optimizer.zero_grad()")
    ap.add_argument("--no-eval", action="store_true", help="THÍ NGHIỆM 2: bỏ model.eval() khi đánh giá")
    ap.add_argument("--tag", default="", help="hậu tố cho tên file lưu (vd: nozg)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    data = prepare_data(args.data)
    train_loader, val_loader, test_loader = make_loaders(data, args.batch_size)

    model = MLP(data["n_features"]).to(device)
    print(f"device={device} | n_features={data['n_features']} | tham số={count_params(model):,}")
    criterion = make_criterion(data["y_train"], device)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)

    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = ROOT / "models" / f"best{suffix}.pt"
    t0 = time.time()
    hist = train(model, optimizer, criterion, train_loader, val_loader, device, epochs=args.epochs,
                 zero_grad=not args.no_zero_grad, use_eval=not args.no_eval,
                 patience=args.patience, ckpt=ckpt)
    print(f"Thời gian train: {time.time() - t0:.1f}s")

    plot_learning_curve(hist, ROOT / "reports" / f"learning_curve{suffix}.png")

    model.load_state_dict(torch.load(ckpt, map_location=device))      # nạp bản tốt nhất
    _, test_pr = evaluate(model, test_loader, criterion, device)
    print(f"Test PR-AUC (checkpoint tốt nhất): {test_pr:.4f}")
    print(f"Đã lưu: {ckpt} và {ROOT / 'reports' / f'learning_curve{suffix}.png'}")


if __name__ == "__main__":
    main()