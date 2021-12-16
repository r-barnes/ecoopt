import cvxpy as cp
import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
from Piecewise2DConvex import Piecewise2DConvex

a1: float = 2
a2: float = 60
b1: float = 0.5
b2: float = 2
L: float = 1
W: float = 1

def gfunc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
  return 1/(a1/L/np.power(x1, b1) + a2/W/np.power(x2, b2))

xvalues = np.linspace(0.01, 20, 20)
yvalues = np.linspace(0.01, 20, 20)
# xx, yy = np.meshgrid(xvalues, yvalues)
# zz = gfunc(xx, yy)

# pts = np.vstack((
#   xx.ravel(), yy.ravel(), zz.ravel()
# )).T

# hull = sp.spatial.ConvexHull(pts)

# triangles = hull.points[hull.simplices]

# centroids = np.sum(triangles, axis=1)/3

# z_of_pt = gfunc(centroids[:,0], centroids[:,1])
# good_triangles = ~(centroids[:,2] < z_of_pt)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# for s in hull.simplices[hull.equations[:,2]>0]:
#     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#     ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

# plt.show()



p2d = Piecewise2DConvex(gfunc, xvalues, yvalues)

# p2d.plot_full_comparison()
p2d.plot_sampled_comparison(100)








# z = cp.Variable()
# xp = cp.Parameter()
# yp = cp.Parameter()
# constraints = []
# for eq in hull.equations[hull.equations[:,2]>0]:
#   constraints.append(eq[0]*xp+eq[1]*yp+eq[2]*z+eq[3]<=0)
# objective = cp.Maximize(z)
# prob = cp.Problem(objective, constraints)

# results = []
# for xval in xvalues:
#   for yval in yvalues:
#     actual = gfunc(xval, yval)
#     xp.value = xval
#     yp.value = yval
#     optval = prob.solve()
#     results.append((actual, z.value))
#     print((actual, z.value))

# results = np.array(results)



# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(pts[:,0], pts[:,1], results[:,0])
# ax.scatter(pts[:,0], pts[:,1], results[:,1])
# for s in hull.simplices[hull.equations[:,2]>0]:
#     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#     ax.plot(pts[s, 1], pts[s, 0], pts[s, 2], "r-")


# plt.show()


# abs_diff = np.abs(results[:,0] - results[:,1])

# rel_diff = np.abs(results[:,0] - results[:,1])/results[:,0]

# print("Maximum abs difference: ", np.max(abs_diff))
# print("Maximum relative difference: ", np.max(rel_diff))
# print("Mean abs difference", np.mean(abs_diff))
# print("Mean relative difference: ", np.mean(rel_diff))