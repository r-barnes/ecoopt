#!/usr/bin/env python3

import matplotlib.pyplot as plt

from src.Iwasa2000_when_flower import Iwasa2000_when_flower
from src.MirmiraniOster1978_single_season import MirmiraniOster1978
from src.MirmiraniOster1978_two_species_single_season import MirmiraniOster1978_TwoSpeciesSingleSeason

p = Iwasa2000_when_flower()
fig = p.plotVariables()
fig.savefig("imgs/iwasa2000_when_flower.pdf")

# p = MirmiraniOster1978()
# fig = p.plotVariables()
# fig.savefig("imgs/MirmiraniOster1978_single_season.pdf")

# p = MirmiraniOster1978_TwoSpeciesSingleSeason()
# fig = p.plotVariables()
# fig.show()
# plt.show()