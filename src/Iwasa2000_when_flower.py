from .problem import cp, Problem

def Iwasa2000_when_flower(T: float = 8.0, dt: float = 0.05) -> Problem:
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt)

  a = p.add_parameter("a", value=0.1)
  h = p.add_parameter("h", value=1.0)
  F0 = p.add_parameter("F0", value=0.5)

  u = p.add_control_var("u", dim=2, lower_bound=0)
  F = p.add_time_var("F", lower_bound=0, initial=F0)
  R = p.add_time_var("R", lower_bound=0, initial=0)
  g = p.add_time_var("g", lower_bound=0, anchor_last=True)

  for _, ti in p.time_indices():
    p.michaelis_menten_constraint(g[ti], F[ti], β1=h, β2=1.0, β3=a) # TODO: Check a-h ordering
    p.constrain_control_sum_at_time(u, g[ti], ti)
    p.dconstraint(F, ti, u[ti,0])
    p.dconstraint(R, ti, u[ti,1])

  optval = p.solve(cp.Maximize(R[-1]))

  return p