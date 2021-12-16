from .problem import cp, Problem

def MirmiraniOster1978(T: float = 8.0, dt: float = 0.05) -> Problem:
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt)

  rhat   = p.add_parameter("rhat", value=0.5)
  rtilde = p.add_parameter("rtilde", value=0.2)
  μ      = p.add_parameter("μ", value=0.1)
  ν      = p.add_parameter("ν", value=0.1)
  P0     = p.add_parameter("P0", value=0.05)

  u = p.add_control_var("u", dim=2, lower_bound=0)
  P = p.add_time_var("P", lower_bound=0, initial=P0)
  S = p.add_time_var("S", lower_bound=0, initial=0)

  for ti in p.time_indices():
    p.constrain_control_sum_at_time(u, P[ti], ti)
    p.dconstraint(P, ti, rhat   * u[ti,0] - μ * P[ti])
    p.dconstraint(S, ti, rtilde * u[ti,1] - ν * S[ti])

  optval = p.solve(cp.Maximize(S[-1]))

  return p