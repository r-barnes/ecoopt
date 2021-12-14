from .problem import Problem, Maximize

def Iwasa1989_multi_season(
  a: float = 0.2, b: float = 1, f: float = 0.05, h: float = 0.05,
  T: float = 100.0, S0: float = 2, dt: float = 0.5, years: int = 8, γ: float = 0.8
):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt, years=years, seasonize=True)

  u = p.add_control_var("u", dim=1, lower_bound=0, upper_bound=None)
  F = p.add_time_var("F", lower_bound=0, upper_bound=None, initial=0)
  S = p.add_time_var("S", lower_bound=0, upper_bound=None, initial=S0)
  g = p.add_time_var("g", lower_bound=0, upper_bound=None, anchor_last=True)
  R = p.add_year_var("R", lower_bound=0, upper_bound=None)

  for n, t in p.time_indices():
    p.constraint(F[n,t+1] == F[n,t] + p.dt * u[n,t,0])
    p.constraint(S[n,t+1] == S[n,t] + p.dt * (g[n,t] - u[n,t,0]))
    p.michaelis_menten_constraint(g[n,t], F[n,t], β1=f, β2=1.0, β3=h)
    p.constraint(u[n,t,0]<=a*F[n,t]+b)

  for n in p.year_indices():
    p.constraint(F[n, 0] == 0)
    p.constraint(R[n] <= S[n,-1])
    if n < p.years-1:
      p.constraint(S[n+1, 0] == γ * (S[n,-1] - R[n]))

  optval = p.solve(Maximize(p.time_discount("R", 0.85)), solver="ECOS")

  return p