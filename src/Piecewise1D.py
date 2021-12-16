#!/usr/bin/env python3

import concurrent.futures as future
import itertools
from sys import _xoptions
from typing import Any, Callable, Final, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.index_tricks import nd_grid
from tqdm import tqdm

def FitPiecewise1D(
  func: Callable[[np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  xvar: cp.Variable,
):
  """
  Fits a piecewise MIP to a possibly non-convex 1D function using methods from
  doi:10.1016/j.orl.2009.09.005
  """
  n: Final[int] = len(xvalues) # Number of x values

  y_values = func(xvalues)

  constraints = []

  ###### x interpolation ######

  alpha = cp.Variable(n)               # Top RHS paragraph, p. 39
  constraints.append(cp.sum(alpha)==1) # Equation 6
  constraints.append(0<=alpha)         # Top RHS paragraph, p. 39
  constraints.append(alpha<=1)         # Top RHS paragraph, p. 39

  # Make alpha SOS2
  # TODO: h variables are unnecessary if we have SOS2 constraints available
  h = cp.Variable(n-1, boolean=True)   # Top RHS paragraph, p. 39
  constraints.append(cp.sum(h)==1)     # Equation 4
  for i in range(n):                   # Equation 5 + top RHS paragraph
    constraints.append(alpha[i]<=(0 if i-1<0 else h[i-1]) + (0 if i==n-1 else h[i]))

  # Equation 7
  constraints.append(xvar==cp.sum(cp.multiply(alpha, xvalues)))

  fa = cp.Variable()
  fa = cp.sum(cp.multiply(alpha, y_values))

  return fa, constraints

def SolvePiecewise1D(
  func: Callable[[np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  xvalue: float,
  solver: str = "CBC",
  verbose: bool = False,
) -> float:
  """Solve the MIP for the point <xvalue>"""
  xvar = cp.Variable()

  fa, mip_constraints = FitPiecewise1D(func, xvalues, xvar)

  mip_constraints.append(xvar==xvalue)

  problem = cp.Problem(cp.Maximize(1), mip_constraints)
  optval = problem.solve(solver=solver, verbose=verbose)

  return fa.value

def SolvePiecewise1D_partial(args):
  return SolvePiecewise1D(*args)

class Piecewise1D:
  def __init__(
    self,
    func: Callable[[np.ndarray], np.ndarray],
    xvalues: np.ndarray,
    solver: str = "CBC"
  ) -> None:
    self.func = func
    self.xvalues = xvalues.copy()
    self.solver = solver

    self.approx_values = None

  def fit(self, xvar: cp.Variable) -> Tuple[cp.Variable, List[Any]]:
    return FitPiecewise1D(self.xvalues, xvar)

  def _get_approx_values(self) -> np.ndarray:
    """
    Gets the approximated value for each value in <xvalues, yvalues>
    This can take quite a while!
    """
    if self.approx_values is not None:
      return self.approx_values

    approx = np.zeros(len(self.xvalues)) - 9999
    args = ((self.func, self.xvalues, xv, self.solver) for xv in self.xvalues)
    with future.ProcessPoolExecutor() as executor:
      results = np.array(list(tqdm(
        executor.map(
          SolvePiecewise1D_partial,
          args
        ),
        total=len(self.xvalues)
      )))

    return results

  @property
  def absdiff(self) -> float:
    approx = self._get_approx_values()
    actual = self.func(self.xvalues)
    return np.abs(actual - approx)

  @property
  def reldiff(self) -> float:
    approx = self._get_approx_values()
    actual = self.func(self.xvalues)
    return np.abs(actual - approx)/actual

  def plot_full_comparison(self) -> None:
    """
    Compares actual to fitted function at every point on the grid
    """
    zz_approx = self._get_approx_values()

    actual = self.func(self.xvalues)
    abs_diff = np.abs(actual - zz_approx)
    rel_diff = np.abs(actual - zz_approx)/actual

    print("Maximum abs difference: ", np.max(abs_diff))
    print("Maximum relative difference: ", np.max(rel_diff))
    print("Mean abs difference", np.mean(abs_diff))
    print("Mean relative difference: ", np.mean(rel_diff))

    fig = plt.figure()
    axes = []

    axes.append(fig.add_subplot(221))
    axes[-1].plot(self.xvalues, actual, '-')
    axes[-1].title.set_text('Actual Values')

    axes.append(fig.add_subplot(222))
    axes[-1].plot(self.xvalues, zz_approx, '-')
    axes[-1].title.set_text('Fitted Values')

    axes.append(fig.add_subplot(223))
    axes[-1].plot(self.xvalues, abs_diff, '-')
    axes[-1].title.set_text('Abs Diff')

    axes.append(fig.add_subplot(224))
    axes[-1].plot(self.xvalues, rel_diff, '-')
    axes[-1].title.set_text('Rel Diff')

    plt.show()

  def plot_sampled_comparison(self, N: int) -> None:
    pts_x = np.random.uniform(low=min(self.xvalues), high=max(self.xvalues), size=N)

    # Test all the points
    args = ((self.func, self.xvalues, xv, self.solver) for xv in pts_x)
    with future.ProcessPoolExecutor() as executor:
      approx_values = np.array(list(tqdm(
        executor.map(
          SolvePiecewise1D_partial,
          args
        ),
        total=N
      )))

    actual_values = self.func(pts_x)
    abs_diff = np.abs(actual_values - approx_values)
    rel_diff = np.abs(actual_values - approx_values)/actual_values

    print("Maximum abs difference: ", np.max(abs_diff))
    print("Maximum relative difference: ", np.max(rel_diff))
    print("Mean abs difference", np.mean(abs_diff))
    print("Mean relative difference: ", np.mean(rel_diff))

    fig = plt.figure()
    axes = []

    axes.append(fig.add_subplot(221))
    axes[-1].plot(self.xvalues, actual_values, '-')
    axes[-1].title.set_text('Actual Values')

    axes.append(fig.add_subplot(222))
    axes[-1].plot(self.xvalues, approx_values, '-')
    axes[-1].title.set_text('Fitted Values')

    axes.append(fig.add_subplot(223))
    axes[-1].plot(self.xvalues, abs_diff, '-')
    axes[-1].title.set_text('Abs Diff')

    axes.append(fig.add_subplot(224))
    axes[-1].plot(self.xvalues, rel_diff, '-')
    axes[-1].title.set_text('Rel Diff')

    plt.show()
