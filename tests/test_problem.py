import math
import sys
import unittest
from cvxpy.settings import OPTIMAL_INACCURATE
from hypothesis import given
from hypothesis import strategies as st
from unittest.case import SkipTest

import cvxpy as cp
import numpy as np
from src.problem import Problem, Maximize
from src.Piecewise1D import Piecewise1D
from src.Piecewise2D import Piecewise2D


class TestBase(unittest.TestCase):
  def assertFloatClose(self, x: float, y: float, rel_tol: float = 1e-2, abs_tol: float = 0.05) -> None:
    if math.isclose(x, y, abs_tol=abs_tol):
      return
    raise AssertionError(f"FloatClose issue: {x}!~={y}")

class ProblemTests(TestBase):
  @given(
    β1 = st.floats(min_value=0, max_value=10),
    β2 = st.floats(min_value=0, max_value=10),
    β3 = st.floats(min_value=0.1, max_value=10),
    xval = st.floats(min_value=0, max_value=10),
  )
  def test_michaelis_menten_constraint(self, β1: float, β2: float, β3: float, xval: float) -> None:
    def _ref(x: float, β1: float, β2: float, β3: float) -> float:
      return β1*x/(β2+β3*x)

    p = Problem(tmin=0, tmax=10, desired_tstep=1)

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    p.constraint(x==xval)
    p.michaelis_menten_constraint(y, x, β1, β2, β3)

    status, optval = p.solve(Maximize(y), solver="ECOS")

    self.assertEqual(status, cp.OPTIMAL)
    self.assertFloatClose(y.value, _ref(xval, β1, β2, β3))

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

    status, optval = p.solve(Maximize(w))

    print(f"Hyper {w.value} <= {x.value} * {y.value} <= {x.value*y.value}")

    # self.assertEqual(status, cp.OPTIMAL)
    # self.assertFloatClose(optval, _ref(xmax, ymax))


def nonconvex_1d_func(x: np.ndarray) -> np.ndarray:
  return np.sin(x)


def nonconvex_2d_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  a1 = 2
  a2 = 60
  b1 = 0.5
  b2 = 2
  L = 1
  W = 1
  return 1/( a1/L/np.power(x, b1) + a2/W/np.power(y, b2) )


class Piecewise1DTests(unittest.TestCase):
  @unittest.skip("demonstrating skipping")
  def test_piecewise_func(self):
    xvalues = np.arange(0, 10, 0.1)

    p1d = Piecewise1D(
      func=nonconvex_1d_func,
      xvalues = xvalues,
    )

    self.assertTrue(np.max(p1d.absdiff) < 2e-10)


class Piecewise2DTests(unittest.TestCase):
  @unittest.skip("demonstrating skipping")
  def test_piecewise_func(self):
    xvalues = np.arange(0.01, 6, 0.3)
    yvalues = np.arange(0.01, 5, 0.3)

    p2d = Piecewise2D(
      func=nonconvex_2d_func,
      xvalues = xvalues,
      yvalues = yvalues,
    )

    self.assertTrue(np.max(p2d.absdiff) < 0.05)
