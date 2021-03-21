import math
import unittest
from cvxpy.settings import OPTIMAL_INACCURATE
from hypothesis import given
from hypothesis import strategies as st

import cvxpy as cp
from src.problem import Problem


class TestBase(unittest.TestCase):
  def assertFloatClose(self, x: float, y: float, rel_tol: float = 1e-2, abs_tol: float = 0.05) -> None:
    if math.isclose(x, y, abs_tol=abs_tol):
      return
    raise AssertionError(f"FloatClose issue: {x}!~={y}")

class ProblemTests(TestBase):
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

    p.constraint(x == xmax)
    p.michaelis_menten_constraint(y, x, β1, β2, β3)

    status, optval = p.solve(cp.Maximize(y))

    if 0 <= xmax - _ref(xmax, β1, β2, β3):
      self.assertEqual(status, cp.OPTIMAL)
      self.assertFloatClose(optval, _ref(xmax, β1, β2, β3))
    else:
      # y>x in michaelis_menten_constraint
      # TODO: Way to work around this?
      pass

  @given(
    xmax = st.floats(min_value=0.1, max_value=10),
    ymax = st.floats(min_value=0.1, max_value=10),
  )
  def test_hyperbolic_constraint(self, xmax: float, ymax: float) -> None:
    def _ref(x: float, y: float) -> float:
      return math.sqrt(x * y)

    p = Problem(tmin=0, tmax=10, desired_tstep=1)

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    w = cp.Variable()

    p.constraint(x <= xmax)
    p.constraint(y <= ymax)
    p.hyperbolic_constraint(w, x, y)

    status, optval = p.solve(cp.Maximize(w))

    self.assertEqual(status, cp.OPTIMAL)
    self.assertFloatClose(optval, _ref(xmax, ymax))