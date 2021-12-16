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

def FitPiecewise2DMIP(
  func: Callable[[np.ndarray, np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  yvalues: np.ndarray,
  xvar: cp.Variable,
  yvar: cp.Variable,
  big_M: Optional[float] = None,
):
  """
  Fits a piecewise MIP to a possibly non-convex 2D function using methods from
  doi:10.1016/j.orl.2009.09.005
  """
  m: Final[int] = len(yvalues) # Number of y transects
  n: Final[int] = len(xvalues) # Number of x values in each y transect

  zz = func(*np.meshgrid(xvalues, yvalues))

  if big_M is None:
    big_M = 10000  # TODO: Come up with something better
  assert big_M is not None

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

  # Top RHS paragraph, p. 40
  # beta_j takes the value 1 iff the given value y-bar belongs to [y_j, y_{j+1})
  beta = cp.Variable(m - 1, boolean=True)

  # Equation 11
  # Makes beta into an SOS1
  constraints.append(cp.sum(beta)==1)

  # Equation 9
  constraints.append(yvar <= cp.sum(cp.multiply(beta,yvalues[1:])))
  # Equation 10
  constraints.append(yvar >= cp.sum(cp.multiply(beta,yvalues[:-1])))

  fa = cp.Variable()
  for j in range(m-1):
    # Equation 12
    constraints.append(fa <= cp.sum(cp.multiply(alpha, zz[j,:])) + big_M * (1-beta[j]))
    # Equation 13
    constraints.append(fa >= cp.sum(cp.multiply(alpha, zz[j,:])) - big_M * (1-beta[j]))

  return fa, constraints

def SolvePiecewise2DMIP(
  func: Callable[[np.ndarray, np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  yvalues: np.ndarray,
  xvalue: float,
  yvalue: float,
  big_M: Optional[float] = None,
  solver: str = "CBC",
  verbose: bool = False,
) -> float:
  """Solve the MIP for the point <xvalue, yvalue>"""
  xvar = cp.Variable()
  yvar = cp.Variable()

  fa, mip_constraints = FitPiecewise2DMIP(func, xvalues, yvalues, xvar, yvar, big_M)

  mip_constraints.append(xvar==xvalue)
  mip_constraints.append(yvar==yvalue)

  problem = cp.Problem(cp.Maximize(1), mip_constraints)
  optval = problem.solve(solver=solver, verbose=verbose)

  return fa.value

def SolvePiecewise2DMIP_partial(args):
  return SolvePiecewise2DMIP(*args)

class Piecewise2DMIP:
  def __init__(
    self,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xvalues: np.ndarray,
    yvalues: np.ndarray,
    big_M: Optional[float] = None,
    solver: str = "CBC"
  ) -> None:
    self.func = func
    self.xvalues = xvalues.copy()
    self.yvalues = yvalues.copy()
    self.big_M = big_M
    self.solver = solver

    self.approx_values = None

  def fit(self, xvar: cp.Variable, yvar: cp.Variable) -> Tuple[cp.Variable, List[Any]]:
    return FitPiecewise2DMIP(self.func, self.xvalues, self.yvalues, xvar, yvar, self.big_M)

  def _get_approx_values(self) -> np.ndarray:
    """
    Gets the approximated value for each value in <xvalues, yvalues>
    This can take quite a while!
    """
    if self.approx_values is not None:
      return self.approx_values

    approx = np.zeros((len(self.yvalues), len(self.xvalues))) - 9999
    values = itertools.product(self.xvalues, self.yvalues)
    args = ((self.func, self.xvalues, self.yvalues, xv, yv, self.big_M, self.solver) for xv, yv in values)
    with future.ProcessPoolExecutor() as executor:
      results = list(tqdm(
        executor.map(
          SolvePiecewise2D_partial,
          args
        ),
        total=len(self.xvalues)*len(self.yvalues)
      ))

    for xi in range(len(self.xvalues)):
      for yi in range(len(self.yvalues)):
          approx[yi, xi] = results.pop(0)

    self.approx_values = approx

    return approx

  @property
  def absdiff(self) -> float:
    zz_approx = self._get_approx_values()
    xx, yy = np.meshgrid(self.xvalues, self.yvalues)
    zz_actual = self.func(xx, yy)
    return np.abs(zz_actual - zz_approx)

  @property
  def reldiff(self) -> float:
    zz_approx = self._get_approx_values()
    xx, yy = np.meshgrid(self.xvalues, self.yvalues)
    zz_actual = self.func(xx, yy)
    return np.abs(zz_actual - zz_approx)/zz_actual

  def plot_full_comparison(self) -> None:
    """
    Compares actual to fitted function at every point on the grid
    """
    zz_approx = self._get_approx_values()

    xx, yy = np.meshgrid(self.xvalues, self.yvalues)
    zz_actual = self.func(xx, yy)

    abs_diff = np.abs(zz_actual - zz_approx)

    rel_diff = np.abs(zz_actual - zz_approx)/zz_actual

    print("Maximum abs difference: ", np.max(abs_diff))
    print("Maximum relative difference: ", np.max(rel_diff))
    print("Mean abs difference", np.mean(abs_diff))
    print("Mean relative difference: ", np.mean(rel_diff))

    fig = plt.figure()
    axes = []

    axes.append(fig.add_subplot(221, projection='3d'))
    axes[-1].plot_surface(xx, yy, zz_actual)
    axes[-1].title.set_text('Actual Values')

    axes.append(fig.add_subplot(222, projection='3d'))
    axes[-1].plot_surface(xx, yy, zz_approx)
    axes[-1].title.set_text('Fitted Values')

    axes.append(fig.add_subplot(223, projection='3d'))
    axes[-1].plot_surface(xx, yy, abs_diff)
    axes[-1].title.set_text('Abs Diff')

    axes.append(fig.add_subplot(224, projection='3d'))
    axes[-1].plot_surface(xx, yy, rel_diff)
    axes[-1].title.set_text('Rel Diff')

    def on_move(event):
      for ax in axes:
        if event.inaxes == ax:
          for oax in axes:
            if ax==oax:
              continue
            oax.view_init(elev=ax.elev, azim=ax.azim)
          break
      else:
        return
      fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

  def plot_sampled_comparison(self, N: int) -> None:
    pts_x = np.random.uniform(low=min(self.xvalues), high=max(self.xvalues), size=(N,1))
    pts_y = np.random.uniform(low=min(self.yvalues), high=max(self.yvalues), size=(N,1))
    pts = np.hstack((pts_x, pts_y))

    # Test all the points
    args = ((self.func, self.xvalues, self.yvalues, xv, yv, self.big_M, self.solver) for xv, yv in zip(pts_x, pts_y))
    with future.ProcessPoolExecutor() as executor:
      approx_values = np.array(list(tqdm(
        executor.map(
          SolvePiecewise2D_partial,
          args
        ),
        total=N
      )))

    actual_values = self.func(pts[:,0], pts[:,1])

    abs_diff = np.abs(actual_values - approx_values)

    rel_diff = np.abs(actual_values - approx_values)/actual_values

    print("Maximum abs difference: ", np.max(abs_diff))
    print("Maximum relative difference: ", np.max(rel_diff))
    print("Mean abs difference", np.mean(abs_diff))
    print("Mean relative difference: ", np.mean(rel_diff))

    fig = plt.figure()
    axes = []

    axes.append(fig.add_subplot(221, projection='3d'))
    axes[-1].scatter(pts[:,0], pts[:,1], actual_values)
    axes[-1].title.set_text('Actual Values')

    axes.append(fig.add_subplot(222, projection='3d'))
    axes[-1].scatter(pts[:,0], pts[:,1], approx_values)
    axes[-1].title.set_text('Fitted Values')

    axes.append(fig.add_subplot(223, projection='3d'))
    axes[-1].scatter(pts[:,0], pts[:,1], abs_diff)
    axes[-1].title.set_text('Abs Diff')

    axes.append(fig.add_subplot(224, projection='3d'))
    axes[-1].scatter(pts[:,0], pts[:,1], rel_diff)
    axes[-1].title.set_text('Rel Diff')

    def on_move(event):
      for ax in axes:
        if event.inaxes == ax:
          for oax in axes:
            if ax==oax:
              continue
            oax.view_init(elev=ax.elev, azim=ax.azim)
          break
      else:
        return
      fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()
