#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from sklearn.preprocessing import normalize #TODO: Remove this and use my own normalize function to save dependency
from .Piecewise1D import Piecewise1D
from .Piecewise2DMIP import Piecewise2DMIP
from .Piecewise2DConvex import Piecewise2DConvex

# import torch.nn as nn

# from learn_to_separate import SeparatorNet

index3d_type = Tuple[int, Union[int,slice], Union[int,slice]]

def Minimize(x: Expression):
  return cp.Minimize(x)

def Maximize(x: Expression):
  return cp.Maximize(x)

def gen_timeseries(start: float, stop: float, timestep: float) -> Tuple[np.ndarray, float]:
  """Return evenly spaced values within a given half-open interval [start, stop)

  Returns: (Timeseries, Actual Timestep)
  """
  num = int(round((stop-start)/timestep))
  timeseries = np.linspace(start=start, stop=stop, num=num)
  actual_timestep = timeseries[1] - timeseries[0]
  return timeseries, actual_timestep


class Variable3D:
    def __init__(self, shape: Tuple[int, int, int], name: str, *args: Any, **kwargs: Any) -> None:
      self.args = args
      self.kwargs = kwargs
      if len(shape)!=3:
        raise RuntimeError("Variable3D can only have 3 dimensions")
      self.years = shape[0]
      self.var_shape = shape[1:]
      if self.years<1:
        raise RuntimeError("Number of years must be >0")
      init_var = lambda name: Variable(self.var_shape, name, *args, **kwargs)
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
    def __getitem__(self, index: index3d_type) -> Variable:
      if len(index)==2:
        index = (0,) + index
      self._check_season_bounds(index)
      return self.data[index[0]][index[1:]]
    def __iter__(self) -> Iterator[Variable]:
      return (x for x in self.data.values())
    def items(self) -> Iterator[Tuple[int,Variable]]:
      return ((n,x) for n, x in self.data.items())


class Problem:
  def __init__(self, tmin: float, tmax: float, desired_tstep: float, years: int = 1, seasonize: bool = False) -> None:
    ts, dt = gen_timeseries(start=tmin, stop=tmax, timestep=desired_tstep)
    self.vars: Dict[str, Variable] = {}
    self.controls: Dict[str, Variable3D] = {}
    self.constraints: List[Constraint] = []
    self.parameters: Dict[str, cp.Parameter] = {}
    self.timeseries: np.ndarray = ts
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

  def tfirst(self, var: str) -> Variable:
    if self.years==1 and not self.seasonize:
      return self.vars[var][0]
    else:
      return self.vars[var][0,0]

  def tlast(self, var: str) -> Variable:
    if self.years==1 and not self.seasonize:
      return self.vars[var][-1]
    else:
      return self.vars[var][-1,-1]

  def add_time_var(
    self,
    name: str,
    initial: Optional[Union[float, cp.Parameter]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    anchor_last: bool = False
  ) -> Variable:
    """[summary]

    Args: TODO
        anchor_last - The last value of the variable is forced equal to zero

    Returns:
        [type]: [description]
    """
    is_pos = lower_bound is not None and lower_bound >= 0

    self.vars[name] = Variable(
      self._time_shape(),
      name=name,
      pos=is_pos, # cvxpy gains extra analysis powers if pos is used
    )
    self.vars[name].ts_type = "time"

    if lower_bound is not None:
      self.constraint(lower_bound<=self.vars[name])
    if upper_bound is not None:
      self.constraint(self.vars[name]<=upper_bound)
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
    initial: Optional[Union[float, cp.Parameter]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    anchor_last: bool = False
  ) -> Variable:
    self.vars[name] = Variable(
      self.years,
      name=name,
      pos=(lower_bound>=0) # cvxpy gains extra analysis powers if pos is used
    )
    self.vars[name].ts_type = "year"

    if lower_bound is not None:
      self.constraint(lower_bound<=self.vars[name])
    if upper_bound is not None:
      self.constraint(self.vars[name]<=upper_bound)
    if initial is not None:
      self.constraint(self.vars[name][0]==initial)
    if anchor_last:
      self.constraint(self.vars[name][-1]==0)

    return self.vars[name]

  def add_parameter(self, name: str, value: Optional[Union[float, np.ndarray]] = None) -> cp.Parameter:
    if isinstance(value, np.ndarray):
      self.parameters[name] = cp.Parameter(shape=value.shape, name=name)
    else:
      self.parameters[name] = cp.Parameter(name=name)

    self.parameters[name].value = value

    return self.parameters[name]

  def add_control_var(self,
    name: str,
    dim: int,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
  ) -> Variable3D:
    self.controls[name] = Variable3D(
      (self.years, len(self.timeseries), dim),
      name=name,
      pos=(lower_bound>=0)
    )
    self.controls[name].ts_type = "control"

    for x in self.controls[name]:
      if lower_bound is not None:
        self.constraint(lower_bound<=x)
      if upper_bound is not None:
        self.constraint(x<=upper_bound)

    return self.controls[name]

  def dconstraint(
    self,
    var: Variable,
    t: Union[int, Tuple[int, int]],
    rhs: Expression,
  ) -> None:
    if isinstance(t, tuple):
      self.constraints.append(
        var[t[0],t[1]+1] == var[t] + self.dt * rhs
      )
    elif isinstance(t, int):
      self.constraints.append(
        var[t+1] == var[t] + self.dt * rhs
      )

  def constraint(self, constraint: Constraint) -> None:
    self.constraints.append(constraint)

  def add_piecewise_function(
    self,
    func: Callable[[np.ndarray], float],
    yvar: Variable,
    xvar: Variable,
    xmin: float,
    xmax: float,
    n: int,
    last_ditch: Optional[float] = None,
    use_sos2: bool = False,
  ) -> Variable:
    #William, p. 149
    xvals = np.linspace(start=xmin, stop=xmax, num=n)
    if last_ditch is not None:
      xvals = np.hstack((xvals, last_ditch))
    yvals = func(xvals)
    l = Variable(len(xvals))
    self.constraint(xvar==xvals*l)
    self.constraint(yvar==yvals*l)
    self.constraint(cp.sum(l)==1)
    self.constraint(l>=0)
    if use_sos2:
      self.add_sos2_constraint(l)
    return yvar

  def add_piecewise1d_function(
    self,
    func: Piecewise1D,
    xvar: Variable,
  ) -> Variable:
    fitted_y, constraints = func.fit(xvar)
    self.constraints.extend(constraints)
    return fitted_y

  def add_piecewise2d_function(
    self,
    func: Union[Piecewise2DMIP, Piecewise2DConvex],
    xvar: Variable,
    yvar: Variable,
  ) -> Variable:
    fitted_z, constraints = func.fit(xvar, yvar)
    self.constraints.extend(constraints)
    return fitted_z

  def michaelis_menten_constraint(
    self,
    lhs_var: Variable,
    rhs_var: Variable,
    β1: float = 1,
    β2: float = 1,
    β3: float = 1,
  ) -> None:
    """lhs_var <= β1*rhs_var/(β2+β3*rhs_var)"""
    β1 = β1 / β3
    β2 = β2 / β3
    self.constraint(lhs_var <= β1 * (1-β2*cp.inv_pos(β2+rhs_var)))

    # self.constraint(0<=rhs_var - lhs_var) #TODO: Is there any way to get rid of this constraint?
    # self.constraint(cp.SOC(
    #   β1 * β2  +  β1 * rhs_var  -  β2 * lhs_var,
    #   cp.vstack([
    #     β1 * rhs_var,
    #     β2 * lhs_var,
    #     β1 * β2
    #   ])
    # ))

  def hyperbolic_constraint(self, w: Variable, x: Variable, y: Variable) -> None:
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
      elif var.ts_type=="y":
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

  def _error_on_bad_status(self, status: Optional[str]) -> None:
    if status == "infeasible":
      raise RuntimeError("Problem was infeasible!")
    elif status == "unbounded":
      raise RuntimeError("Problem was unbounded!")
    elif status == "unbounded_inaccurate":
      raise RuntimeError("Problem was unbounded and inaccurate!")

  def solve(
    self,
    objective,
    solver: str = "CBC",
    **kwargs: Any,
  ) -> Tuple[Union[str,None],float]:
    problem = cp.Problem(objective, self.constraints)

    print("Problem is DCP?", problem.is_dcp())

    optval = problem.solve(solver, **kwargs)

    self._error_on_bad_status(problem.status)

    return problem.status, optval

  def time_indices(self) -> Iterator[Tuple[int, int]]:
    if self.years==1 and not self.seasonize:
      return ((0,t) for t in range(len(self.timeseries)-1))
    else:
      return ((y,t) for y in range(self.years) for t in range(len(self.timeseries)-1))

  def idx2time(self, time_index: int) -> float:
    assert 0<=time_index<len(self.timeseries)
    return self.timeseries[time_index]

  def year_indices(self) -> Iterator[int]:
    return range(self.years)

  def time_discount(self, var: str, σ: float) -> Expression:
    expr: Expression = 0
    for n in range(self.years):
      expr += self.vars[var][n] * (σ**n)
    return expr

  def constrain_control_sum_at_time(
    self,
    control: Variable3D,
    sum_var: Expression,
    t: int,
    n: int = 0
  ) -> None:
    self.constraint(cp.sum(control[n, t, :]) <= sum_var) #TODO: Should this be == or <= ?

  def add_sos2_constraint(self, x: Variable) -> None:
    # TODO: From https://www.philipzucker.com/trajectory-optimization-of-a-pendulum-with-mixed-integer-linear-programming/
    assert len(x.shape) == 1
    n = x.size
    z = Variable(n - 1, boolean=True)
    self.constraint(0 <= x)
    self.constraint(x[0] <= z[0])
    self.constraint(x[-1] <= z[-1])
    self.constraint(x[1:-1] <= z[:-1]+z[1:])
    self.constraint(cp.sum(z) == 1)
    self.constraint(cp.sum(x) == 1)

  # def add_learned_separable_constraint(
  #   self,
  #   net: SeparatorNet,
  #   inp: Variable,
  #   use_sos2: bool = False
  # ) -> Variable:
  #   # TODO: Writeup
  #   assert len(inp.shape) == 1
  #   assert inp.size == net.input_dim

  #   outputs = [inp]

  #   for layer in net.sequential:
  #     if isinstance(layer, nn.Linear):
  #       weight = layer.weight.detach().cpu().numpy()
  #       bias = layer.bias.detach().cpu().numpy()
  #       this_output = Variable(weight.shape[0])
  #       self.constraint(this_output == weight @ outputs[-1] + bias)
  #       outputs.append(this_output)
  #     elif isinstance(layer, nn.Tanh):
  #       assert len(outputs[-1].shape) == 1
  #       this_output = Variable(outputs[-1].size)
  #       for y, x in zip(this_output, outputs[-1]):
  #         self.add_piecewise_function(np.tanh, y, x, -1, 10, 50, 100, use_sos2=use_sos2)
  #       outputs.append(this_output)

  #   assert outputs[-1].size == 1

  #   return outputs[-1]