import cvxpy as cp
import numpy as np
import torch

from problem import Problem
from learn_to_separate import learn_to_separate


def func(x: np.ndarray) -> np.ndarray:
  assert len(x.shape)==2
  a1 = 2
  a2 = 60
  b1 = 0.5
  b2 = 2
  L = 1
  W = 1
  return 1/(a1/L/x[:,0]**b1 + a2/W/x[:,1]**b2)


N = 1000
data = np.random.uniform(low=0, high=20, size=(N,2)).astype(np.float32)
yvals = func(data)

model = learn_to_separate(data, yvals, [5], show_plot=True)

breakpoint()


p = Problem(tmin=0, tmax=10, desired_tstep=1)
x = cp.Variable(2, pos=True)
y = p.add_learned_separable_constraint(model, x, use_sos2=True)
p.constraint(x[0] <= 0.6)
p.constraint(x[1] <= 0.6)
status, optval = p.solve(cp.Maximize(y))

print(f"{status} -> {optval}")
print("Torch prediction", model(torch.tensor(x.value).float().cuda()))

# self.assertEqual(status, cp.OPTIMAL)
# self.assertFloatClose(optval, _ref(xmax, ymax))
