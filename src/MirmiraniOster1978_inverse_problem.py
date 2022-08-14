import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pycombina
from numpy.lib.function_base import diff

from src.problem import Minimize
from src.MirmiraniOster1978_single_season import MirmiraniOster1978

np.random.seed(123490)

# Solve the original system
problem = MirmiraniOster1978()
problem.solve()

# Get a copy of original result for plotting
P_original, S_original = problem["P"].value.copy(), problem["S"].value.copy()

# Take a subset of the points and add noise to them
P_with_noise = P_original + np.random.normal(
  scale=0.05, size=P_original.shape
)
sample = np.random.choice(
  list(range(P_original.shape[0])), replace=False, size=10
)

# Solve problem with noise
diff_orig_noise = cp.norm(P_with_noise[sample] - problem["P"][sample])
problem.objective(Minimize(diff_orig_noise))
problem.solve(solver="ECOS")

# Plot
fig = problem.plotVariables(hide_vars=["f"], norm_controls=True)
fig.get_axes()[0].plot(problem.timeseries, P_original, '-', label="Actual P")
fig.get_axes()[0].plot(problem.timeseries, S_original, '-', label="Actual S")
fig.get_axes()[0].scatter(problem.timeseries[sample], P_with_noise[sample])
fig.legend().remove()
fig.legend()
plt.show()
fig.savefig("imgs/Mirmirani_inverse_bad_fit.pdf", bbox_inches='tight')



# Solve problem with noise and no constraints on the control
# This is a "relaxed" version of the problem we want to solve
problem.objective(Minimize(diff_orig_noise))
problem.solve(solver="ECOS")

# Use Combinatorial Integer Approximation to find the best control strategy
# given the relaxed solution
ba = pycombina.BinApprox(
  problem.timeseries,
  problem.get_normalized_control_value("u")[:-1,:]
)
ba.set_n_max_switches([1,1])

milp = pycombina.CombinaMILP(ba)
milp.solve()

# Now, resolve the problem, but constraining the control based on the CIA
problem = MirmiraniOster1978(constrain=True)
problem.objective(Minimize(diff_orig_noise))
problem.constraint(problem["u"][0,:-1,0] == cp.multiply(ba.b_bin.T[:,0], problem["P"][:-1]))
problem.constraint(problem["u"][0,:-1,1] == cp.multiply(ba.b_bin.T[:,1], problem["P"][:-1]))
problem.solve(solver="ECOS")
#TODO: Abstract the above to something beautiful

# Plot
fig = problem.plotVariables(hide_vars=["f"], norm_controls=True)
fig.get_axes()[0].plot(problem.timeseries, P_original, '-', label="Actual P")
fig.get_axes()[0].plot(problem.timeseries, S_original, '-', label="Actual S")
fig.get_axes()[0].scatter(problem.timeseries[sample], P_with_noise[sample])
fig.legend()
fig.savefig("imgs/Mirmirani_inverse_good_fit.pdf", bbox_inches='tight')