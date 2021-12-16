from .problem import cp, Problem, Piecewise2DConvex

import numpy as np

def Iwasa2000_shoot_root_balance(
  T: float = 100,
  X10: float = 1,
  X20: float = 1,
  a1: float = 2,
  a2: float = 60,
  b1: float = 0.5,
  b2: float = 2,
  L: float = 1,
  W: float = 1,
  dt: float = 0.2
):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt)

  u  = p.add_control_var("u", dim=3, lower_bound=0)
  X1 = p.add_time_var("X1", lower_bound=0, initial=X10)
  X2 = p.add_time_var("X2", lower_bound=0, initial=X20)
  R  = p.add_time_var("R",  lower_bound=0, initial=0)

  print("Building problem...")
  for _, ti in p.time_indices():
    g = p.add_piecewise2d_function(gpiecewise, X1[ti], X2[ti])
    p.constrain_control_sum_at_time(u, g, ti)
    p.dconstraint(X1, ti, u[ti, 0])
    p.dconstraint(X2, ti, u[ti, 1])
    p.dconstraint(R,  ti, u[ti, 2])

  print("Solving...")
  optval = p.solve(cp.Maximize(R[-1]), verbose=True)

  return p