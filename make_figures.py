#!/usr/bin/env python3

import matplotlib.pyplot as plt

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

# p = Iwasa2000_when_flower()
# fig = p.plotVariables()
# fig.savefig("imgs/iwasa2000_when_flower.pdf")

# p = MirmiraniOster1978()
# fig = p.plotVariables()
# fig.show()
# plt.show()
# fig.savefig("imgs/MirmiraniOster1978_single_season.pdf")

# p = MirmiraniOster1978_TwoSpeciesSingleSeason()
# fig = p.plotVariables()
# fig.show()
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