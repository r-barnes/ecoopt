#!/usr/bin/env python3

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


def gen_timeseries(start: float, stop: float, timestep: float) -> Tuple[List[float], float]:
  """Return evenly spaced values within a given half-open interval [start, stop)

  Returns: (Timeseries, Actual Timestep)
  """
  num = int(round((stop-start)/timestep))
  timeseries = list(np.linspace(start=start, stop=stop, num=num))
  actual_timestep = timeseries[1] - timeseries[0]
  return timeseries, timestep


class Problem:
  def __init__(self, tmin: float, tmax: float, desired_tstep: float) -> None:
    ts, dt = gen_timeseries(start=tmin, stop=tmax, timestep=desired_tstep)
    self.vars: Dict[str, cp.Variable] = {}
    self.controls: Dict[str, cp.Variable] = {}
    self.constraints = []
    self.timeseries: List[float] = ts
    self.dt: float = dt

  def add_time_var(self,
    name: str,
    initial: Optional[float] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    anchor_last: bool = False
  ) -> None:
    self.vars[name] = cp.Variable(
      len(self.timeseries),
      name=name,
      pos=(lb>=0) # cvxpy gains extra analysis powers if pos is used
    )

    if lb is not None:
      self.constraint(lb<=self.vars[name])
    if ub is not None:
      self.constraint(self.vars[name]<=ub)
    if initial is not None:
      self.constraint(self.vars[name][0]==initial)
    if anchor_last:
      self.constraint(self.vars[name][-1]==0)

    return self.vars[name]

  def add_control_var(self,
    name: str,
    dim: int,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
  ) -> None:
    # No control on the last timestep
    self.controls[name] = cp.Variable(
      (len(self.timeseries)-1, dim),
      name=name,
      pos=(lb>=0) # cvxpy gains extra analysis powers if pos is used
    )

    if lb is not None:
      self.constraint(lb<=self.controls[name])
    if ub is not None:
      self.constraint(self.controls[name]<=ub)

    return self.controls[name]

  def constraint(self, constraint) -> None:
    self.constraints.append(constraint)

  # TODO: p.addPiecewiseFunction(lambda x: x**0.75, yvar=g[t], xvar=F[t], xmin=0, xmax=30, n=50, last_ditch=80)
  def addPiecewiseFunction(self,
    func: Callable[[np.ndarray], float],
    yvar: cp.Variable,
    xvar: cp.Variable,
    xmin: float,
    xmax: float,
    n: int,
    last_ditch: Optional[float] = None,
  ) -> cp.Variable:
    #William, p. 149
    xvals = np.linspace(start=xmin, stop=xmax, num=n)
    if last_ditch is not None:
      xvals = np.hstack((xvals, last_ditch))
    yvals = func(xvals)
    l = cp.Variable(len(xvals))
    self.constraint(xvar==xvals*l)
    self.constraint(yvar==yvals*l)
    self.constraint(cp.sum(l)==1)
    self.constraint(l>=0)
    return yvar

  def michaelis_menten_constraint(
    self,
    lhs_var: cp.Variable,
    rhs_var: cp.Variable,
    β1: float = 1,
    β2: float = 1,
    β3: float = 1,
  ) -> None:
    """lhs_var <= β1*X/(β2+β3*X)"""
    β1 = β1 / β3
    β2 = β2 / β3
    self.constraint(0<=rhs_var - lhs_var) #TODO: Is there any way to get rid of this constraint?
    self.constraint(cp.SOC(
      β1 * β2  +  β1 * rhs_var  -  β2 * lhs_var,
      cp.vstack([
        β1 * rhs_var,
        β2 * lhs_var,
        β1 * β2
      ])
    ))

  def hyperbolic_constraint(self, w: cp.Variable, x: cp.Variable, y: cp.Variable) -> None:
    """dot(w,w)<=x*y"""
    self.constraint(cp.SOC(x + y, cp.vstack([2 * w, x - y])))

  def plotVariables(self, norm_controls: bool = True) -> plt.Figure:
    fig, axs = plt.subplots(2)
    for x in self.vars:
      axs[0].plot(self.timeseries, self.vars[x].value, label=x)
    for x in self.controls:
      val = self.controls[x].value.copy()
      if norm_controls:
        val = normalize(val, axis=1, norm='l2')
      for ci in range(self.controls[x].shape[1]):
        axs[1].plot(self.timeseries[:-1], val[:,ci], label=f"{x}_{ci}")
    fig.legend()
    return fig

  def solve(self, objective, verbose: bool = False) -> Tuple[Union[str,None],float]:
    problem = cp.Problem(objective, self.constraints)
    optval = problem.solve("SCS", verbose=verbose)
    return problem.status, optval

  def time_indices(self) -> Generator[int, None, None]:
    return range(len(self.timeseries)-1)

  def constrain_control_sum_at_time(self, control: cp.Variable, t: int, sum_val) -> None:
    self.constraint(cp.sum(control[t, :]) == sum_val)
