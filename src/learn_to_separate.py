#!/usr/bin/env python3

from typing import Generator, List, Sequence, Tuple, TypeVar
import itertools

import cvxpy as cp
from cvxpy.problems.objective import Minimize
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler


T = TypeVar("T")

def _pairwise(iterable: Sequence[T]) -> Generator[Tuple[T, T], None, None]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class SeparatorNet(nn.Module):
  def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
    super().__init__()
    assert isinstance(hidden_dims, list)
    assert len(hidden_dims) >= 0

    self.input_dim = input_dim

    layers = nn.ModuleList()
    for prev_dim, next_dim in _pairwise([input_dim] + hidden_dims + [1]):
      layers.append(nn.Linear(prev_dim, next_dim))
      layers.append(nn.Tanh())
    del layers[-1] # Don't want a tanh activation on that last layer!
    self.sequential = nn.Sequential(*layers)

    self.x_transform_model = MinMaxScaler()
    self.y_transform_model = MinMaxScaler()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.sequential(x)

  def x_transform(self, xvals: np.ndarray) -> np.ndarray:
    return self.x_transform_model.fit_transform(xvals)

  def y_transform(self, yvals: np.ndarray) -> np.ndarray:
    return self.y_transform_model.fit_transform(yvals.reshape(-1, 1))


def learn_to_separate(
  xvals: np.ndarray,
  yvals: np.ndarray,
  hidden_layers: List[int],
  use_gpu: bool = True,
  show_plot: bool = False
) -> SeparatorNet:
  if len(xvals.shape) != 2:
    raise RuntimeError("xvals must be 2D")
  if len(yvals.shape) != 1:
    raise RuntimeError("yvals must be 1D")

  device = "cpu"
  if use_gpu:
    if not torch.cuda.is_available():
      raise RuntimeWarning("use_gpu was specified, but no GPU is available. Falling back to CPU")
    else:
      device = torch.device("cuda:0")

  model = SeparatorNet(xvals.shape[1], hidden_layers).to(device)
  xvals = model.x_transform(xvals)
  yvals = model.y_transform(yvals)

  learning_rate = 0.1
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.MSELoss()

  xvals = torch.tensor(xvals).requires_grad_().to(device)
  yvals = torch.tensor(yvals).requires_grad_().to(device)

  iter = 0
  num_epochs = 1000
  batch_size = 100
  for epoch in range(num_epochs):
    sample_idx = np.random.randint(low=0, high=len(xvals), size=batch_size)
    batch = xvals[sample_idx,:]
    batch_vals = yvals[sample_idx]

    optimizer.zero_grad()  # Clear gradients w.r.t. parameters
    outputs = model(batch) # Forward pass to get output
    loss = criterion(outputs, batch_vals)
    loss.backward()        # Getting gradients w.r.t. parameters
    optimizer.step()       # Updating parameters

    if epoch % 20 == 0:
      loss = F.mse_loss(model(xvals), yvals)
      print(f"Loss at {epoch}: {loss}")

  if show_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pdata = xvals.detach().cpu().numpy()
    pfvals = yvals.detach().cpu().numpy()
    pmfvals = model(xvals).detach().cpu().numpy()
    ax.scatter(pdata[:,0], pdata[:,1], pfvals, cmap="plasma")
    ax.scatter(pdata[:,0], pdata[:,1], pmfvals, cmap="plasma")
    plt.show()

  return model