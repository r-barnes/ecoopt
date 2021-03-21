from .problem import cp, Problem


def MirmiraniOster1978_TwoSpeciesSingleSeason(r=0.5, T=1.0, P0=0.05, desired_dt=0.01):
  p = Problem(tmin=0.0, tmax=T, desired_tstep=desired_dt)

  u1 = p.add_control_var("u1", dim=2, lb=0, ub=None)
  u2 = p.add_control_var("u2", dim=2, lb=0, ub=None)

  print(repr(u1))

  P1 = p.add_time_var("P1", lb=0, ub=None, initial=P0)
  S1 = p.add_time_var("S1", lb=0, ub=None, initial=0)

  P2 = p.add_time_var("P2", lb=0, ub=None, initial=P0)
  S2 = p.add_time_var("S2", lb=0, ub=None, initial=0)

  for t in p.time_indices():
    g1 = cp.Variable(2, pos=True, name="g1")
    g2 = cp.Variable(2, pos=True, name="g2")

    # p.constraint(g1 <= 10) #TODO: Remove
    # p.constraint(g2 <= 10) #TODO: Remove
    a = cp.Variable(pos=True, name="a")
    b = cp.Variable(pos=True, name="b")
    p.constraint(r - P1[t] == a)
    p.constraint(r - P2[t] == b)

    p.hyperbolic_constraint(g1[0], r - P2[t], u1[t, 0])
    p.hyperbolic_constraint(g2[0], r - P1[t], u2[t, 0])

    p.hyperbolic_constraint(g1[1], r - P2[t], u1[t, 1])
    p.hyperbolic_constraint(g2[1], r - P1[t], u2[t, 1])

    p.constraint(P1[t+1] == P1[t] + p.dt * g1[0])
    p.constraint(P2[t+1] == P2[t] + p.dt * g2[0])

    p.constraint(S1[t+1] == S1[t] + p.dt * g1[1])
    p.constraint(S2[t+1] == S2[t] + p.dt * g2[1])

    p.constrain_control_sum_at_time(u1, t, P1[t])
    p.constrain_control_sum_at_time(u2, t, P2[t])

  status, optval = p.solve(cp.Maximize(S1[-1]+S2[-1]))

  print("Status",status)
  print("Optval",optval)

  return p
