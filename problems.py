#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

def gen_timeseries(start: float, stop: float, timestep: float) -> Tuple[List[float], float]:
  """Return evenly spaced values within a given half-open interval [start, stop)

  Returns: (Timeseries, Actual Timestep)
  """
  num = int(round((stop-start)/timestep))
  timeseries = list(np.linspace(start=start, stop=stop, num=num))
  actual_timestep = timeseries[1] - timeseries[0]
  return timeseries, timestep


def timevar(timeseries: List[float], name: Optional[str] = None):
  """Generate a variable with a value for each point in the timeseries"""
  return cp.Variable(len(timeseries), name=name)


def MichaelisMentenConstraint(lhs_var: cp.Variable, rhs_var: cp.Variable, beta1: float, beta2: float) -> cp.SOC:
  """lhs_var <= B1*X/(B2+X)"""
  return cp.SOC(
    beta1 * beta2  +  beta1 * rhs_var  -  beta2 * lhs_var,
    cp.vstack([
      beta1 * rhs_var,
      beta2 * lhs_var,
      beta1 * beta2
    ])
  )


# def HyperbolicConstraint(w: cp.Variable, x: cp.Variable, y: cp.Variable) -> cp.SOC:
#   """dot(w,w)<=x*y"""
#   return cp.SOC(x+y, cp.vstack([2*w, x-y]))


class Problem:
  def __init__(self):
    self.vars: Dict[str, cp.Variable] = {}
    self.controls: Dict[str, cp.Variable] = {}
    self.constraints = []
    self.timeseries: List[float] = []
    self.dt: float = 0

  def setTimeSeries(self, start: float, stop: float, timestep: float):
    self.timeseries, self.dt = gen_timeseries(start=start, stop=stop, timestep=timestep)

  def addTimeVar(self,
    name: str,
    initial: Optional[float] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None
  ) -> None:
    self.vars[name] = cp.Variable(len(self.timeseries), name=name)
    if lb is not None:
      self.addConstraint(lb<=self.vars[name])
    if ub is not None:
      self.addConstraint(self.vars[name]<=ub)
    if initial is not None:
      self.addConstraint(self.vars[name][0]==initial)

    return self.vars[name]

  def addTimeControlVar(self,
    name: str,
    dim: int,
    lb: Optional[float] = None,
    ub: Optional[float] = None
  ) -> None:
    self.controls[name] = cp.Variable((len(self.timeseries), dim), name=name)
    if lb is not None:
      self.addConstraint(lb<=self.controls[name])
    if ub is not None:
      self.addConstraint(self.controls[name]<=ub)

    return self.controls[name]

  def addConstraint(self, constraint) -> None:
    self.constraints.append(constraint)

  def addPiecewiseFunction(self,
    func,
    xvar: cp.Variable,
    xmin: float,
    xmax: float,
    n: int,
    last_ditch: Optional[float] = None,
    yvar: Optional[cp.Variable] = None,
  ) -> cp.Variable:
    #William, p. 149
    xvals = np.linspace(start=xmin, stop=xmax, num=n)
    if last_ditch is not None:
      xvals = np.hstack((xvals, last_ditch))

    yvals = func(xvals)
    if yvar is None:
      yvar  = cp.Variable()

    l = cp.Variable(len(xvals))
    self.addConstraint(xvar==xvals*l)
    self.addConstraint(yvar==yvals*l)
    self.addConstraint(cp.sum(l)==1)
    self.addConstraint(l>=0)
    return yvar

  def plotVariables(self) -> plt.Figure:
    fig = plt.figure()
    for x in self.vars:
      plt.plot(self.timeseries, self.vars[x].value, label=x)
    for x in self.controls:
      for ci in range(self.controls[x].shape[1]):
        plt.plot(self.timeseries, self.controls[x].value[:,ci], label=f"{x}_{ci}")
    plt.legend()
    plt.show()

  def solve(self, objective, verbose: bool = False) -> float:
    problem = cp.Problem(objective, self.constraints)
    optval = problem.solve("ECOS", verbose=verbose)
    return optval


def Iwasa2000_when_flower(a=0.1, h=1, T=8.0, F0=0.5, desired_dt=0.1):
  p = Problem()
  p.setTimeSeries(0.0, T, desired_dt)

  u = p.addTimeControlVar("u", dim=2, lb=0, ub=None)
  F = p.addTimeVar("F", lb=0, ub=None, initial=F0)
  R = p.addTimeVar("R", lb=0, ub=None, initial=0)
  g = p.addTimeVar("g", lb=0, ub=None)

  for i, t in enumerate(p.timeseries):
    if i == len(p.timeseries) - 1:
      p.addConstraint(u[-1,:]==0)
      p.addConstraint(g[-1]==0)
      break

    # p.addConstraint(MichaelisMentenConstraint(g[i], F[i], a, 1))
    this_g = p.addPiecewiseFunction(lambda x: a*x**0.75, xvar=F[i], xmin=0, xmax=30, n=50, last_ditch=80, yvar=g[i])
    p.addConstraint(cp.sum(u[i,:])==this_g)
    p.addConstraint(F[i+1] == F[i] + u[i,0] * p.dt)
    p.addConstraint(R[i+1] == R[i] + u[i,1] * p.dt)

  optval = p.solve(cp.Maximize(R[-1]), verbose=True)

  print(f"Optval: {optval}")

  p.plotVariables()

  return optval


Iwasa2000_when_flower()