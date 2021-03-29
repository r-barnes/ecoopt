from .problem import cp, Problem

def Iwasa1989_multi_season(a=0.2, b=1, f=0.05, h=0.05, T=100.0,
                           S0=2, desired_dt=0.5, years=8, γ=0.8):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=desired_dt, years=years, seasonize=True)

  u = p.add_control_var("u", dim=1, lb=0, ub=None)
  F = p.add_time_var("F", lb=0, ub=None, initial=0)
  S = p.add_time_var("S", lb=0, ub=None, initial=S0)
  g = p.add_time_var("g", lb=0, ub=None, anchor_last=True)
  R = p.add_year_var("R", lb=0, ub=None)

  for n, t in p.time_indices():
    p.constraint(F[n,t+1] == F[n,t] + p.dt * u[n,t,0])
    p.constraint(S[n,t+1] == S[n,t] + p.dt * (g[n,t] - u[n,t,0]))
    # TODO: Check a-h ordering
    p.michaelis_menten_constraint(g[n,t], F[n,t], β1=f, β2=1.0, β3=h)
    p.constraint(u[n,t,0]<=a*F[n,t]+b)

  for n in p.year_indices():
    p.constraint(F[n, 0] == 0)
    p.constraint(R[n] <= S[n,-1])
    if n < p.years-1:
      p.constraint(S[n+1, 0] == γ * (S[n,-1] - R[n]))

  optval = p.solve(cp.Maximize(p.time_discount("R", 0.85)))

  return p