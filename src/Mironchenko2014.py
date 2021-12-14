import math

from .problem import Problem, Maximize, Variable

def MironchenkoFigure4() -> Problem:
  def zeta(t: float) -> float:
    return 0.2 + 0.8 * abs(math.sin(math.pi / 12 * t))

  def mu(t: float) -> float:
    return 0.4*abs(math.cos(math.pi / 12 *t))

  def g(x: Variable):
    return 2.5 * x

  omega = 0.1

  p = Problem(tmin=0.0, tmax=60, desired_tstep=0.1)

  u  = p.add_control_var("u", dim=2, lower_bound=0, upper_bound=None)
  # Initial state extracted graphically from Figure 4
  x1 = p.add_time_var("x1", lower_bound=0, upper_bound=None, initial=0)
  x2 = p.add_time_var("x2", lower_bound=0, upper_bound=None, initial=0)
  x3 = p.add_time_var("x3", lower_bound=0, upper_bound=None, initial=0.2949852507374615)
  f  = p.add_time_var("f",  lower_bound=0, upper_bound=None, anchor_last=True)

  for _, ti in p.time_indices():
    t = p.idx2time(ti)
    p.michaelis_menten_constraint(f[ti], x1[ti], β1=0.5, β2=1, β3=0.01)
    p.constrain_control_sum_at_time(u, g(x1[ti]), ti)
    p.dconstraint(x1, ti, p.dt, u[ti,0] - mu(t) * x1[ti])
    p.dconstraint(x2, ti, p.dt, u[ti,1])
    p.dconstraint(x3, ti, p.dt, zeta(t) * f[ti] - u[ti,0] - u[ti,1] - omega * x3[ti])

  status, optval = p.solve(Maximize(x2[-1]), solver="ECOS", verbose=True)

  print(status)

  return p