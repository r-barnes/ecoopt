#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from src.problem import Maximize

from src.Iwasa2000_when_flower import Iwasa2000_when_flower
from src.MirmiraniOster1978_single_season import MirmiraniOster1978
from src.MirmiraniOster1978_two_species_single_season import MirmiraniOster1978_TwoSpeciesSingleSeason
from src.Iwasa1989_multi_season import Iwasa1989_multi_season
from src.Mironchenko2014_figure2 import MironchenkoFigure2
from src.Mironchenko2014_figure3 import MironchenkoFigure3
from src.Mironchenko2014_figure4 import MironchenkoFigure4
from src.Mironchenko2014_figure5a import MironchenkoFigure5a
from src.Mironchenko2014_figure5c import MironchenkoFigure5c
from src.Mironchenko2014_figure5e import MironchenkoFigure5e
from src.Iwasa2000_shoot_root_balance import Iwasa2000_shoot_root_balance
from src.Piecewise2DConvex import Piecewise2DConvex

# p = Iwasa2000_when_flower()
# fig = p.plotVariables()
# fig.savefig("imgs/iwasa2000_when_flower.pdf")

# p = MirmiraniOster1978()
# fig = p.plotVariables()
# fig.show()
# plt.show()
# fig.savefig("imgs/MirmiraniOster1978_single_season.pdf")


uval = None
while True:
  p = MirmiraniOster1978_TwoSpeciesSingleSeason()
  if uval is not None:
    p.constraint(p.controls["u2"][0,:,:]==uval)

  # The problem here is figuring what to optimize and how to find an ESS with linear programming?
  status, optval = p.solve(Maximize(p.vars["S1"][-1]), solver="ECOS", verbose=True)

  print("Status", status)
  print("Optval", optval)

  fig = p.plotVariables()
  fig.show()
  plt.show()

  uval = p.controls["u1"][0,:,:].value


# p = MirmiraniOster1978_TwoSpeciesSingleSeason()
# fig = p.plotVariables()
# fig.show()
# plt.show()

# plt.plot(p.vars["P1"].value, p.vars["P2"].value)
# plt.plot(p.vars["S2"].value, p.vars["S1"].value)
# plt.show()

# p = Iwasa1989_multi_season(years=4)
# fig = p.plotVariables(hide_vars=["g"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/Iwasa1989_multi_season.pdf")

# p = MironchenkoFigure2(is_single=True)
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/Mironchenko_figure2a.pdf")

# p = MironchenkoFigure2(is_single=False)
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/Mironchenko_figure2b.pdf")

# Replicates
# p = MironchenkoFigure4()
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/Mironchenko_figure4.pdf")

# Doesn't reproduce
# p = MironchenkoFigure3(is_annual=True)
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/Iwasa1989_multi_season.pdf")

# Replicates
# p = MironchenkoFigure5a()
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/MironchenkoFigure5a.pdf")

# Replicates
# p = MironchenkoFigure5c()
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/MironchenkoFigure5c.pdf")

# Nearly replicates
# p = MironchenkoFigure5e()
# fig = p.plotVariables(hide_vars=["f"], norm_controls=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/MironchenkoFigure5e.pdf")

# def gfunc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
#   a1 = 2
#   a2 = 60
#   b1 = 0.5
#   b2 = 2
#   L = 1
#   W = 1
#   return 1/(a1/L/np.power(x1, b1) + a2/W/np.power(x2, b2))

# gpiecewise = Piecewise2DConvex(
#   func=gfunc,
#   xvalues=np.linspace(0.01, 20, 20),
#   yvalues=np.linspace(0.01, 20, 20),
# )

# fig = gpiecewise.plot_samples()
# fig.show()
# plt.show()
# fig.savefig("imgs/iwasa1984_hull_samples.pdf", bbox_inches='tight')

# fig = gpiecewise.plot_hull(full=True)
# fig.show()
# plt.show()
# fig.savefig("imgs/iwasa1984_hull_full.pdf", bbox_inches='tight')

# fig = gpiecewise.plot_hull(full=False)
# fig.show()
# plt.show()
# fig.savefig("imgs/iwasa1984_hull_upper.pdf", bbox_inches='tight')




# p = Iwasa2000_shoot_root_balance()
# fig = p.plotVariables(hide_vars=["g"], norm_controls=True)
# fig.show()
# plt.show()