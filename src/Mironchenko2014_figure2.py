import math

from .problem import Problem, Maximize, Variable

def MironchenkoFigure2(is_stable: bool) -> Problem:
  """Values and equations are drawn from Mironchenko's source code TODO"""
  def zeta(t: float) -> float:
    return 0.2 + 0.8 * abs(math.sin(math.pi / 12 * t))

  def mu(t: float) -> float:
    if is_stable:
      return 0.8 * abs(math.cos(math.pi / 12 * t))
    else:
      return 1.8 * abs(math.cos(math.pi * t))

  def omega(t: float) -> float:
    return 0.05 if is_stable else t / (1 + 0.5 * t)

  def g(x: Variable):
    return 5 * x if is_stable else 16 * x

  seed_mass = 0.3

  p = Problem(tmin=0.0, tmax=12, desired_tstep=0.02)

  u  = p.add_control_var("u", dim=2, lower_bound=0)
  x1 = p.add_time_var("x1", lower_bound=0, initial=0.05*seed_mass)
  x2 = p.add_time_var("x2", lower_bound=0, initial=0.00*seed_mass)
  x3 = p.add_time_var("x3", lower_bound=0, initial=0.95*seed_mass)
  f  = p.add_time_var("f",  lower_bound=0, anchor_last=True)

  for _, ti in p.time_indices():
    t = p.idx2time(ti)
    p.michaelis_menten_constraint(
      f[ti],  x1[ti], β1=1.5 if is_stable else 4, β2=1.0, β3=0.3
    )
    p.constrain_control_sum_at_time(u, g(x3[ti]), ti)
    p.dconstraint(x1, ti, u[ti,0] - mu(t) * x1[ti])
    p.dconstraint(x2, ti, u[ti,1])
    p.dconstraint(
      x3, ti, zeta(t) * f[ti] - u[ti,0] - u[ti,1] - omega(t) * x3[ti]
    )

  status, optval = p.solve(
    Maximize(x2[-1]), solver="ECOS", verbose=True, max_iters=10000,
    abstol=1e-3, reltol=1e-3, feastol=1e-4
  )

  print(status)

  return p
