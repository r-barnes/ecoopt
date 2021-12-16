#!/usr/bin/env python3

import concurrent.futures as future
import itertools
from scipy.spatial import ConvexHull
from typing import Any, Callable, Final, List, Optional, Tuple
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def _gethull(
  func: Callable[[np.ndarray, np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  yvalues: np.ndarray
):
  xx, yy = np.meshgrid(xvalues, yvalues)
  zz = func(xx, yy)

  pts = np.vstack((
    xx.ravel(), yy.ravel(), zz.ravel()
  )).T

  return pts, ConvexHull(pts)

def FitPiecewise2DConvex(
  func: Callable[[np.ndarray, np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  yvalues: np.ndarray,
  xvar: Variable,
  yvar: Variable,
  maximizing: bool = True,
) -> Tuple[Variable, List[Constraint]]:
  """
  Fits a convex surface hull to the function
  """
  _, hull = _gethull(func, xvalues, yvalues)

  # Variable approximating the function
  fa = Variable()

  if maximizing:
    used_equations = hull.equations[hull.equations[:,2]>0]
  else:
    used_equations = hull.equations[hull.equations[:,2]<0]

  constraints = []
  for eq in used_equations:
    constraints.append(eq[0]*xvar + eq[1]*yvar + eq[2]*fa + eq[3] <= 0)

  return fa, constraints

def SolvePiecewise2DConvex(
  func: Callable[[np.ndarray, np.ndarray], np.ndarray],
  xvalues: np.ndarray,
  yvalues: np.ndarray,
  xvalue: float,
  yvalue: float,
  maximizing: bool = True,
  solver: str = "CBC",
  verbose: bool = False,
) -> float:
  """Solve the MIP for the point <xvalue, yvalue>"""
  xvar = Variable()
  yvar = Variable()

  fa, mip_constraints = FitPiecewise2DConvex(func, xvalues, yvalues, xvar, yvar)

  mip_constraints.append(xvar==xvalue)
  mip_constraints.append(yvar==yvalue)

  problem = cp.Problem(
    cp.Maximize(fa) if maximizing else cp.Minimize(fa),
    mip_constraints
  )
  optval = problem.solve(solver=solver, verbose=verbose)

  return fa.value

def SolvePiecewise2DConvex_partial(kwargs):
  return SolvePiecewise2DConvex(**kwargs)

class Piecewise2DConvex:
  def __init__(
    self,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xvalues: np.ndarray,
    yvalues: np.ndarray,
    maximizing: bool = True,
    solver: str = "CBC",
  ) -> None:
    self.func = func
    self.xvalues = xvalues.copy()
    self.yvalues = yvalues.copy()
    self.maximizing = maximizing
    self.solver = solver

    self.approx_values = None

  def fit(self, xvar: Variable, yvar: Variable) -> Tuple[Variable, List[Any]]:
    return FitPiecewise2DConvex(self.func, self.xvalues, self.yvalues, xvar, yvar, maximizing=self.maximizing)

  def _get_approx_values(self) -> np.ndarray:
    """
    Gets the approximated value for each value in <xvalues, yvalues>
    This can take quite a while!
    """
    if self.approx_values is not None:
      return self.approx_values

    approx = np.zeros((len(self.yvalues), len(self.xvalues))) - 9999
    values = itertools.product(self.xvalues, self.yvalues)
    args = []
    for xv, yv in values:
      args.append({
      "func":self.func,
      "xvalues": self.xvalues,
      "yvalues": self.yvalues,
      "xvalue": xv,
      "yvalue": yv,
      "solver": self.solver,
      "maximizing": self.maximizing
    })
    with future.ProcessPoolExecutor() as executor:
      results = list(tqdm(
        executor.map(
          SolvePiecewise2DConvex_partial,
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

  def plot_samples(self, full: bool = False) -> None:
    pts, hull = _gethull(self.func, self.xvalues, self.yvalues)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=45)
    ax.scatter(pts[:, 0], pts[:,1], pts[:,2])
    return fig

  def plot_hull(self, full: bool = False) -> None:
    pts, hull = _gethull(self.func, self.xvalues, self.yvalues)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=45)
    for s, e in zip(hull.simplices, hull.equations):
      if full or (e[2] > 0 if self.maximizing else e[2] < 0):
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
    return fig

  def plot_sampled_comparison(self, N: int) -> None:
    pts_x = np.random.uniform(low=min(self.xvalues), high=max(self.xvalues), size=(N,1))
    pts_y = np.random.uniform(low=min(self.yvalues), high=max(self.yvalues), size=(N,1))
    pts = np.hstack((pts_x, pts_y))

    # Test all the points
    args = []
    for xv, yv in zip(pts_x, pts_y):
      args.append({
      "func":self.func,
      "xvalues": self.xvalues,
      "yvalues": self.yvalues,
      "xvalue": xv,
      "yvalue": yv,
      "solver": self.solver,
      "maximizing": self.maximizing
    })
    with future.ProcessPoolExecutor() as executor:
      approx_values = np.array(list(tqdm(
        executor.map(
          SolvePiecewise2DConvex_partial,
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
