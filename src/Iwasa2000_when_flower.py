from .problem import cp, Problem

def Iwasa2000_when_flower(
  a: float = 0.1, h: float = 1.0, T: float = 8.0, F0: float = 0.5, dt: float = 0.05
):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt)

  u = p.add_control_var("u", dim=2, lower_bound=0, upper_bound=None)
  F = p.add_time_var("F", lower_bound=0, upper_bound=None, initial=F0)
  R = p.add_time_var("R", lower_bound=0, upper_bound=None, initial=0)
  g = p.add_time_var("g", lower_bound=0, upper_bound=None, anchor_last=True)

  for _, t in p.time_indices():
    p.michaelis_menten_constraint(g[t], F[t], β1=h, β2=1.0, β3=a) # TODO: Check a-h ordering
    p.constrain_control_sum_at_time(u, g[t], t)
    p.constraint(F[t+1] == F[t] + p.dt * u[t,0])
    p.constraint(R[t+1] == R[t] + p.dt * u[t,1])

  optval = p.solve(cp.Maximize(R[-1]))

  return p