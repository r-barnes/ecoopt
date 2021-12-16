from .problem import Problem, Maximize

def Iwasa1989_multi_season(T: float = 100.0, years: int = 8, dt: float=0.5) -> Problem:
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt, years=years, seasonize=True)

  a  = p.add_parameter("a", value = 0.2)
  b  = p.add_parameter("b", value = 1)
  f  = p.add_parameter("f", value = 0.05)
  h  = p.add_parameter("h", value = 0.05)
  S0 = p.add_parameter("S0", value = 2)
  γ  = p.add_parameter("γ", value = 0.8)

  u = p.add_control_var("u", dim=1, lower_bound=0)
  F = p.add_time_var("F", lower_bound=0, initial=0)
  S = p.add_time_var("S", lower_bound=0, initial=S0)
  g = p.add_time_var("g", lower_bound=0, anchor_last=True)
  R = p.add_year_var("R", lower_bound=0)

  for n, ti in p.time_indices():
    p.constrain_control_sum_at_time(u, a*F[n,ti]+b, n=n, t=ti)
    p.dconstraint(F, (n,ti), u[n,ti,0])
    p.dconstraint(S, (n,ti), g[n,ti] - u[n,ti,0])
    p.michaelis_menten_constraint(g[n,ti], F[n,ti], β1=f, β2=1.0, β3=h)

  for n in p.year_indices():
    p.constraint(F[n, 0] == 0)
    p.constraint(R[n] <= S[n,-1])
    if n < p.years-1:
      p.constraint(S[n+1, 0] == γ * (S[n,-1] - R[n]))

  optval = p.solve(Maximize(p.time_discount("R", 0.85)), solver="ECOS")

  return p