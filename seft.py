import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data processing
X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# basic model for speed
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# evaluates a model on validation data
def accuracy(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_val).argmax(dim=1)
        return (preds == y_val).float().mean().item()

# applies a binary mask to model parameters
def apply_mask(model, mask):
    with torch.no_grad():
        for p, m in zip(model.parameters(), mask):
            p *= m

# creates a mask using magnitude pruning
def create_mask(model, sparsity):
    mask = []
    for p in model.parameters():
        tensor = p.detach().cpu().numpy()
        thresh = np.percentile(np.abs(tensor), sparsity * 100)
        m = (np.abs(tensor) > thresh).astype(np.float32)
        mask.append(torch.tensor(m, dtype=torch.float32, device=p.device))
    return mask

# drops weights with the smallest magnitude,
# and grows weights with the largest gradients
def drop_and_grow(model, mask, raw_grads, drop_fraction=0.05):
    new_mask = []

    for p, m, g in zip(model.parameters(), mask, raw_grads):
        # skip biases
        if p.ndim < 2:
            new_mask.append(m.clone())
            continue

        weight = p.detach().cpu().numpy()
        grad = g.detach().cpu().numpy()
        m_np = m.cpu().numpy()

        w_flat = weight.reshape(-1)
        g_flat = grad.reshape(-1)
        m_flat = m_np.reshape(-1)

        active_idx = np.where(m_flat == 1)[0]
        inactive_idx = np.where(m_flat == 0)[0]

        if len(active_idx) == 0:
            new_mask.append(m.clone())
            continue

        # drop the smallest active weights by magnitude
        active_vals = np.abs(w_flat[active_idx])
        drop_k = int(len(active_idx) * drop_fraction)

        drop_mask_flat = np.zeros_like(m_flat, dtype=bool)

        if drop_k > 0:
            drop_order = np.argsort(active_vals)[:drop_k]
            drop_indices = active_idx[drop_order]
            drop_mask_flat[drop_indices] = True
        else:
            drop_indices = np.array([], dtype=int)

        # grow inactive weights with the largest gradient magnitude
        grow_mask_flat = np.zeros_like(m_flat, dtype=bool)
        num_to_grow = len(drop_indices)

        if num_to_grow > 0 and len(inactive_idx) > 0:
            num_to_grow = min(num_to_grow, len(inactive_idx))
            grad_vals = np.abs(g_flat[inactive_idx])
            top_order = np.argsort(grad_vals)[-num_to_grow:]
            top_k = inactive_idx[top_order]
            grow_mask_flat[top_k] = True

        # final mask
        m_new_flat = m_flat.copy()
        m_new_flat[drop_mask_flat] = 0
        m_new_flat[grow_mask_flat] = 1

        m_new = m_new_flat.reshape(m_np.shape)
        new_mask.append(torch.tensor(m_new, dtype=torch.float32, device=p.device))

    return new_mask


def train(model, mask=None, seft=False, epochs=20):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        out = model(X_train)
        loss = loss_fn(out, y_train)

        opt.zero_grad()
        loss.backward()

        raw_grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

        if mask is not None:
            for p, m in zip(model.parameters(), mask):
                if p.grad is not None:
                    p.grad *= m

        opt.step()

        if mask is not None:
            apply_mask(model, mask)

        if seft and mask is not None and epoch > 0 and epoch % 5 == 0:
            mask = drop_and_grow(model, mask, raw_grads, drop_fraction=0.05)
            apply_mask(model, mask)

        print(f"Epoch {epoch} | Loss {loss.item():.4f} | Acc {accuracy(model):.4f}")

    return mask

print("Training dense model...")
model_dense = MLP()
train(model_dense)

print("\nPruning...")
mask = create_mask(model_dense, sparsity=0.8)
apply_mask(model_dense, mask)

print("After pruning acc:", accuracy(model_dense))

print("\nStatic sparse fine-tuning...")
model_static = MLP()
model_static.load_state_dict(model_dense.state_dict())
start = time.perf_counter()
train(model_static, mask=mask, seft=False)
elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.2f} seconds")

print("\nSEFT fine-tuning...")
model_seft = MLP()
model_seft.load_state_dict(model_dense.state_dict())
start = time.perf_counter()
train(model_seft, mask=mask, seft=True)
elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.2f} seconds")