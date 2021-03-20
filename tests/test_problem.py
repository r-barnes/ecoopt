import math
import unittest
from hypothesis import given
from hypothesis import strategies as st

import cvxpy as cp
from src.problem import Problem

class ProblemTests(unittest.TestCase):
  @given(
    β1 = st.floats(min_value=0.1, max_value=10),
    β2 = st.floats(min_value=0.1, max_value=10),
    β3 = st.floats(min_value=0.1, max_value=10),
    xmax = st.floats(min_value=0.1, max_value=10)
  )
  def test_michaelis_menten_constraint(self, β1: float, β2: float, β3: float, xmax: float) -> None:
    def _ref(x: float, β1: float, β2: float, β3: float) -> float:
      return β1*x/(β2+β3*x)

    p = Problem(tmin=0, tmax=10, desired_tstep=1)

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    p.constraint(x <= xmax)
    p.michaelis_menten_constraint(y, x, β1, β2, β3)

    status, optval = p.solve(cp.Maximize(y))

    self.assertEqual(status, cp.OPTIMAL)
    self.assertTrue(math.isclose(optval, _ref(xmax, β1, β2, β3), rel_tol=0.03))

  @given(
    xmax = st.floats(min_value=0.1, max_value=10),
    ymax = st.floats(min_value=0.1, max_value=10),
  )
  def test_michaelis_menten_constraint(self, xmax: float, ymax: float) -> None:
    def _ref(x: float, y: float) -> float:
      return x * y

    p = Problem(tmin=0, tmax=10, desired_tstep=1)

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    w = cp.Variable()

    p.constraint(x <= xmax)
    p.constraint(y <= xmax)
    p.hyperbolic_constraint(w, x, y)

    status, optval = p.solve(cp.Maximize(w))

    self.assertEqual(status, cp.OPTIMAL)
    self.assertTrue(math.isclose(optval, _ref(xmax, ymax), rel_tol=0.03))