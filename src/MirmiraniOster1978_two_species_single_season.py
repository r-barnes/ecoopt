from .problem import Maximize, Problem, Variable

def MirmiraniOster1978_TwoSpeciesSingleSeason(T: float = 10.0, dt: float = 0.1) -> Problem:
  p = Problem(tmin=0.0, tmax=T, desired_tstep=dt)

  r1  = p.add_parameter("r1",  value=0.5)
  r2  = p.add_parameter("r2",  value=0.5)
  P10 = p.add_parameter("P10", value=0.05)
  P20 = p.add_parameter("P20", value=0.05)

  u1 = p.add_control_var("u1", dim=2, lower_bound=0)
  u2 = p.add_control_var("u2", dim=2, lower_bound=0)

  P1 = p.add_time_var("P1", lower_bound=0, initial=P10)
  S1 = p.add_time_var("S1", lower_bound=0, initial=0)

  P2 = p.add_time_var("P2", lower_bound=0, initial=P20)
  S2 = p.add_time_var("S2", lower_bound=0, initial=0)

  for _, ti in p.time_indices():
    dpdt = Variable(2, pos=True, name="g1")
    dsdt = Variable(2, pos=True, name="g2")

    p.hyperbolic_constraint(dpdt[0], r1 - P2[ti], u1[ti, 0])
    p.hyperbolic_constraint(dpdt[1], r2 - P1[ti], u2[ti, 0])

    p.hyperbolic_constraint(dsdt[0], r1 - P2[ti], u1[ti, 1])
    p.hyperbolic_constraint(dsdt[1], r2 - P1[ti], u2[ti, 1])

    p.constraint(r1 - P2[ti] >= 0)
    p.constraint(r2 - P1[ti] >= 0)

    p.dconstraint(P1, ti, dpdt[0])
    p.dconstraint(P2, ti, dpdt[1])

    p.dconstraint(S1, ti, dsdt[0])
    p.dconstraint(S2, ti, dsdt[1])

    p.constrain_control_sum_at_time(u1, P1[ti], ti)
    p.constrain_control_sum_at_time(u2, P2[ti], ti)

  # The problem here is figuring what to optimize and how to find an ESS with linear programming?
  # status, optval = p.solve(Maximize(p.vars["S1"][-1]), solver="ECOS", verbose=True)

  return p