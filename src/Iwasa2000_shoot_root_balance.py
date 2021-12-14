from .problem import cp, Problem

def Iwasa2000_shoot_root_balance(
  T=100,
  X10: float = 1,
  X20: float = 1,
  a1: float = 2,
  a2: float = 60,
  b1: float = 0.5,
  b2: float = 2,
  L: float = 1,
  W: float = 1,
  desired_dt=0.05
):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=desired_dt)

  u = p.add_control_var("u", dim=2, lb=0, ub=None)
  F = p.add_time_var("F", lb=0, ub=None, initial=F0)
  R = p.add_time_var("R", lb=0, ub=None, initial=0)
  g = p.add_time_var("g", lb=0, ub=None, anchor_last=True)

  for t in p.time_indices():
    p.michaelis_menten_constraint(g[t], F[t], β1=h, β2=1.0, β3=a) # TODO: Check a-h ordering
    p.constrain_control_sum_at_time(u, g[t], t)
    p.constraint(F[t+1] == F[t] + p.dt * u[t,0])
    p.constraint(R[t+1] == R[t] + p.dt * u[t,1])

  optval = p.solve(cp.Maximize(R[-1]))

  return p