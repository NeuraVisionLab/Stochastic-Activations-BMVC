"""Quickstart with no dataset download; run: python examples/mlp_quickstart.py"""
import torch
import torch.nn as nn
from stochact import StochasticModel

base = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(inplace=True),
    nn.Linear(32, 3),
)

model = StochasticModel(
    base,
    granularity="node",
    activations="silu",  # default None also means native + SiLU
    p=0.5,
)
model.report()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
inputs = torch.randn(16, 12)
targets = torch.randint(0, 3, (16,))

model.train()
for step in range(3):
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()
    print(f"step={step+1}, loss={loss.item():.4f}")

model.eval()  # BatchNorm/dropout eval mode; StochAct still switches activations.
result = model.predict(inputs, task="classification", n_samples=5)
print("Mean probabilities:", tuple(result.mean.shape))
print("Population variance:", tuple(result.variance.shape))
print("Predictive entropy:", result.predictive_entropy.tolist())
print("Mutual information:", result.mutual_information.tolist())
