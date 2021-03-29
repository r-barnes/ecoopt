#!/usr/bin/env python3

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


index3d_type = Tuple[int, Union[int,slice], Union[int,slice]]


def gen_timeseries(start: float, stop: float, timestep: float) -> Tuple[List[float], float]:
  """Return evenly spaced values within a given half-open interval [start, stop)

  Returns: (Timeseries, Actual Timestep)
  """
  num = int(round((stop-start)/timestep))
  timeseries = list(np.linspace(start=start, stop=stop, num=num))
  actual_timestep = timeseries[1] - timeseries[0]
  return timeseries, timestep


class Variable3D:
    def __init__(self, shape: Tuple[int, int, int], name, *args, **kwargs) -> None:
      self.args = args
      self.kwargs = kwargs
      if len(shape)!=3:
        raise RuntimeError("Variable3D can only have 3 dimensions")
      self.years = shape[0]
      self.var_shape = shape[1:]
      if self.years<1:
        raise RuntimeError("Number of years must be >0")
      init_var = lambda name: cp.Variable(self.var_shape, name, *args, **kwargs)
      self.data = {i:init_var(f"{name}{i}") for i in range(self.years)}
    def __repr__(self) -> str:
      shape = (self.years,) + self.var_shape
      return f"TSVariable({shape}, args={self.args}, kwargs={self.kwargs})"
    def _check_season_bounds(self, index: index3d_type) -> None:
      if len(index)!=3:
        raise IndexError("Insufficient indexing dimensions. Include season?")
      if not isinstance(index[0], int):
        raise IndexError("Season dimension must be an integer!")
      if index[0] not in self.data:
        raise IndexError("Season value is not in the variable's range!")
    def __getitem__(self, index: index3d_type) -> cp.Variable:
      if len(index)==2:
        index = (0,) + index
      self._check_season_bounds(index)
      return self.data[index[0]][index[1:]]
    def __iter__(self) -> Generator[cp.Variable, None, None]:
      return (x for x in self.data.values())
    def items(self) -> Generator[Tuple[int,cp.Variable], None, None]:
      return ((n,x) for n, x in self.data.items())


class Problem:
  def __init__(self, tmin: float, tmax: float, desired_tstep: float, years: int = 1, seasonize: bool = False) -> None:
    ts, dt = gen_timeseries(start=tmin, stop=tmax, timestep=desired_tstep)
    self.vars: Dict[str, cp.Variable] = {}
    self.controls: Dict[str, Variable3D] = {}
    self.constraints = []
    self.timeseries: List[float] = ts
    self.dt: float = dt

    if years<=0:
      raise RuntimeError("Seasons must be >0!")
    if years!=int(years):
      raise RuntimeError("Seasons must be an integer!")
    self.years = years
    self.seasonize = seasonize

  def _time_shape(self) -> Tuple[int ,...]:
    if self.years==1 and not self.seasonize:
      return (len(self.timeseries), )
    else:
      return (self.years, len(self.timeseries))

  def tfirst(self, var: str) -> cp.Variable():
    if self.years==1 and not self.seasonize:
      return self.vars[var][0]
    else:
      return self.vars[var][0,0]

  def tlast(self, var: str) -> cp.Variable():
    if self.years==1 and not self.seasonize:
      return self.vars[var][-1]
    else:
      return self.vars[var][-1,-1]

  def add_time_var(self,
    name: str,
    initial: Optional[float] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    anchor_last: bool = False
  ) -> cp.Variable():
    self.vars[name] = cp.Variable(
      self._time_shape(),
      name=name,
      pos=(lb>=0) # cvxpy gains extra analysis powers if pos is used
    )
    self.vars[name].ts_type = "time"

    if lb is not None:
      self.constraint(lb<=self.vars[name])
    if ub is not None:
      self.constraint(self.vars[name]<=ub)
    if initial is not None:
      self.constraint(self.tfirst(name)==initial)
    if anchor_last:
      if self.years==1 and not self.seasonize:
        self.constraint(self.vars[name][-1]==0)
      else:
        self.constraint(self.vars[name][:,-1]==0)

    return self.vars[name]

  def add_year_var(self,
    name: str,
    initial: Optional[float] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    anchor_last: bool = False
  ) -> cp.Variable():
    self.vars[name] = cp.Variable(
      self.years,
      name=name,
      pos=(lb>=0) # cvxpy gains extra analysis powers if pos is used
    )
    self.vars[name].ts_type = "year"

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
  ) -> Variable3D:
    self.controls[name] = Variable3D(
      (self.years, len(self.timeseries), dim),
      name=name,
      pos=(lb>=0)
    )
    self.controls[name].ts_type = "control"

    for x in self.controls[name]:
      if lb is not None:
        self.constraint(lb<=x)
      if ub is not None:
        self.constraint(x<=ub)

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

  def plotVariables(self, norm_controls: bool = True, hide_vars: Optional[List[str]] = None) -> plt.Figure:
    fig, axs = plt.subplots(2)

    if hide_vars is None:
      hide_vars = []

    for name, var in self.vars.items():
      if name in hide_vars:
        continue
      elif var.ts_type=="time":
        full_times = []
        for n in range(self.years):
          full_times.extend([n*self.timeseries[-1] + x for x in self.timeseries])
        axs[0].plot(full_times, var.value.flatten(), label=name)
      elif var.ts_type=="year":
        full_times = [(n+1)*self.timeseries[-1] for n in range(self.years)]
        axs[0].plot(full_times, var.value.flatten(), '.', label=name)

    full_times = []
    for n in range(self.years):
      full_times.extend([n*self.timeseries[-1] + x for x in self.timeseries])

    for name, var in self.controls.items():
      val = np.vstack([var[n,:,:].value for n in range(self.years)])
      if norm_controls:
        val[val<1e-3] = 0
        val = normalize(val, axis=1, norm='l2')
      for ci in range(val.shape[1]):
        axs[1].plot(full_times, val[:,ci], label=f"{name}_{ci}")
    fig.legend()
    return fig

  def solve(self, objective, verbose: bool = False) -> Tuple[Union[str,None],float]:
    problem = cp.Problem(objective, self.constraints)
    optval = problem.solve("SCS", verbose=verbose)
    return problem.status, optval

  def time_indices(self) -> Generator[Union[int, Tuple[int, int]], None, None]:
    if self.years==1 and not self.seasonize:
      return range(len(self.timeseries)-1)
    else:
      return ((n,t) for n in range(self.years) for t in range(len(self.timeseries)-1))

  def year_indices(self) -> Generator[int, None, None]:
    return range(self.years)

  def time_discount(self, var: str, σ: float):
    expr = 0
    for n in range(self.years):
      expr += self.vars[var][n] * (σ**n)
    return expr

  def constrain_control_sum_at_time(self, control: cp.Variable, sum_val, t: int, n: int = 0) -> None:
    self.constraint(cp.sum(control[n, t, :]) == sum_val)
