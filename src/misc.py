from dataclasses import dataclass
from cvxpy.constraints.constraint import Constraint
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import cvxpy as cp

xL=-4
yL=-4
xU=0.6255960255544074
yU=0.3744039755194276

@dataclass
class McCormickEnvelope:
  w: cp.Variable
  x: cp.Variable
  y: cp.Variable
  xL: float
  xU: float
  yL: float
  yU: float
  def getConstraints(s) -> List[Constraint]:
    return [
      s.w >= s.xL*s.y + s.x*s.yL - s.xL*s.yL,
      s.w >= s.xU*s.y + s.x*s.yU - s.xU*s.yU,
      s.w <= s.xU*s.y + s.x*s.yL - s.xU*s.yL,
      s.w <= s.xL*s.y + s.x*s.yU - s.xL*s.yU
    ]

w = cp.Variable()
x = cp.Variable()
y = cp.Variable()
me = McCormickEnvelope(w, x, y, xL, xU, yL, yU)

constraints = me.getConstraints()
constraints.append(-6*x+8*y<=3)
constraints.append(3*x-y<=3)
constraints.append(x<=1.5)
constraints.append(y<=1.5)
constraints.append(xL<=x)
constraints.append(yL<=y)
constraints.append(x<=4)
constraints.append(y<=4)
objective = cp.Minimize(-x + w - y)
prob = cp.Problem(objective, constraints)
optval = prob.solve()
print("optval",optval)
print("x.value",x.value)
print("y.value",y.value)