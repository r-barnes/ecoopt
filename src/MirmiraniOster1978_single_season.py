from .problem import cp, Problem

def MirmiraniOster1978(rhat=0.5, rtilde=0.2, μ=0.1, ν=0.1, T=8.0, P0=0.05, desired_dt=0.05):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=desired_dt)

  u = p.add_control_var("u", dim=2, lb=0, ub=None)
  P = p.add_time_var("P", lb=0, ub=None, initial=P0)
  S = p.add_time_var("S", lb=0, ub=None, initial=0)

  for t in p.time_indices():
    p.constrain_control_sum_at_time(u, t, P[t])
    p.constraint(P[t+1] == P[t] + p.dt * (rhat   * u[t,0] - μ * P[t]))
    p.constraint(S[t+1] == S[t] + p.dt * (rtilde * u[t,1] - ν * S[t]))

  optval = p.solve(cp.Maximize(S[-1]))

  return p