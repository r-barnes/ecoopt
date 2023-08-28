import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Tuple

def michaelis_menten_constraint(
  lhs_var: cp.Variable,
  rhs_var: cp.Variable,
  β1: float = 1,
  β2: float = 1,
  β3: float = 1,
) -> None:
  """lhs_var <= β1*rhs_var/(β2+β3*rhs_var)"""
  β1 = β1 / β3
  β2 = β2 / β3
  return lhs_var <= β1 * (1-β2*cp.inv_pos(β2+rhs_var))

def gen_timeseries(start: float, stop: float, timestep: float) -> Tuple[npt.NDArray[np.float_], float]:
  """Return evenly spaced values within a given half-open interval [start, stop)

  Returns: (Timeseries, Actual Timestep)
  """
  num = int(round((stop-start)/timestep))
  timeseries = np.linspace(start=start, stop=stop, num=num)
  actual_timestep = timeseries[1] - timeseries[0]
  return timeseries, actual_timestep

a = 0.1
h = 0.5
F0 = 0.5

T = 8.0
desired_dt = 0.05
times, dt = gen_timeseries(start=0.0, stop=T, timestep=desired_dt)
ntimes = len(times)

F1 = cp.Variable(ntimes, pos=True, name="F1")
R1 = cp.Variable(ntimes, pos=True, name="R1")

u1 = cp.Variable(ntimes, pos=True, name="u1")
u2 = cp.Variable(ntimes, pos=True, name="u2")
g = cp.Variable(ntimes, pos=True, name="g")

constraints = [
  F1[0] == F0,
  R1[0] == 0,
]

constraints += [F1[t+1] == F1[t] + dt * u1[t] for t in range(ntimes-1)]
constraints += [R1[t+1] == R1[t] + dt * u2[t] for t in range(ntimes-1)]
constraints += [u1[t] + u2[t] == g[t] for t in range(ntimes)]
objective = cp.Maximize(R1[-1])

F2 = [0 for _ in range(ntimes)]
for fi in range(15):
  if fi != 0:
    F2 = F1.value.copy()
  mmc = [michaelis_menten_constraint(g[t], F1[t], β1=h, β2=1+F2[t], β3=0.1) for t in range(ntimes)]
  problem = cp.Problem(objective, constraints + mmc)
  objective_value = problem.solve(solver=cp.GUROBI)
  print(objective_value)
  if fi == 0:
    F1_original = F1.value.copy()
    F2_original = F2.copy()
    R1_original = R1.value.copy()
    u1_original = u1.value.copy()
    u2_original = u2.value.copy()


fig, axs = plt.subplots(2, 2)

axs[0, 0].set_title("Without Competition")
axs[0, 0].plot(times, u1_original, label="Leaf Control")
axs[0, 0].plot(times, u2_original, label="Seed Control")
axs[0, 0].legend()

axs[0, 1].set_title("With Competition")
axs[0, 1].plot(times, u1.value, label="Leaf Control")
axs[0, 1].plot(times, u2.value, label="Seed Control")
axs[0, 1].legend()

axs[1, 0].plot(times, F1_original, label="Leaves 1")
axs[1, 0].plot(times, R1_original, label="Seeds 1")
axs[1, 0].plot(times, F2_original, label="Leaves 2")
axs[1, 0].legend()

axs[1, 1].plot(times, F1.value, label="Leaves 1")
axs[1, 1].plot(times, R1.value, label="Seeds 1")
axs[1, 1].plot(times, F2, label="Leaves 2")
axs[1, 1].legend()

plt.show()
