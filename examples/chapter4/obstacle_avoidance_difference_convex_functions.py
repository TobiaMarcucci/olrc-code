import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Random obstacles.
radius = .1
centers = np.array([
    [.7, .3],
    [.8, .8],
    [.3, .2],
    [.3, .7],
    [.6, .6],
])

# Initial trajectory.
initial_points = np.array([
    [0, 0],
    [.1, 0],
    [.2, 0],
    [.3, 0],
    [.4, 0],
    [.5, 0],
    [.5, .1],
    [.5, .2],
    [.5, .3],
    [.5, .4],
    [.5, .5],
    [.5, .6],
    [.5, .7],
    [.6, .7],
    [.7, .7],
    [.8, .7],
    [.9, .7],
    [1, .7],
    [1, .8],
    [1, .9],
    [1, 1],
])

# Objective function.
def objective_function(points):
    return cp.sum_squares(points[1:] - points[:-1])

# Linearization of constraints.
def linearized_constraint(p, p_new, c):
    diff = p - c
    dist = np.linalg.norm(diff)
    offset = radius - dist
    gradient = - diff / dist
    return offset + gradient @ (p_new - p) <= 0

# Difference of convex functions.
tol = 1e-3
solutions = [initial_points]
values = [objective_function(initial_points).value]
while True:
    points = solutions[-1]
    new_points = cp.Variable(points.shape)
    constraints = [new_points[0] == points[0], new_points[-1] == points[-1]]
    for p, p_new in zip(points, new_points):
        for c in centers:
            constraints.append(linearized_constraint(p, p_new, c))
    cost = objective_function(new_points)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    if (values[-1] - prob.value) / prob.value < tol:
        break
    solutions.append(new_points.value)
    values.append(prob.value)

# Plot result.
plt.figure()
for c in centers:
    patch = Circle(c, radius, facecolor='lightcoral', edgecolor='black')
    plt.gca().add_patch(patch)
for points, value in zip(solutions, values):
    rounded_value = np.round(value, 5)
    plt.plot(*points.T, marker='o', label=f"Objective value = {rounded_value}")
plt.legend()
plt.show()