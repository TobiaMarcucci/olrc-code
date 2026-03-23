import numpy as np
import matplotlib.pyplot as plt

# Parameters.
m = 1 # mass
g = 10 # gravity
l = 1 # length
h = .0001 # discretization step
T = 5 # time horizon

# Control policy.
def pi(x):
    ### YOUR CODE HERE
    ### YOUR CODE HERE
    ### YOUR CODE HERE
    return 0

# Closed-loop dynamics.
def f(x):
    return np.array([
        x[1],
        (m * g * l * np.sin(x[0]) + pi(x)) / (m * l ** 2)])

# Plot trajectories for different initial angles.
plt.figure()
K = int(T / h)
traj = np.zeros((K, 2))
initial_angles = np.pi * np.arange(11) / 5
for theta_0 in initial_angles:
    traj[0] = [theta_0, 0]
    for k in range(K - 1):
        traj[k + 1] = traj[k] + h * f(traj[k])
    plt.plot(traj[:,0], traj[:,1], label=fr'$\theta_0={np.round(theta_0, 2)}$')

# Plot options.
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot \theta$')
plt.title('Pendulum trajectories')
plt.legend()
plt.grid()
plt.show()