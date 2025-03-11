import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

np.random.seed(42)

means = [np.array([-5, -5]), np.array([5, 5])]
covs = [np.array([[0.2, 0], [0, 0.2]]), np.array([[0.2, 0], [0, 0.2]])]
weights = [0.5, 0.5]

x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

Z = np.zeros_like(X)
for mean, cov, weight in zip(means, covs, weights):
    rv = multivariate_normal(mean, cov)
    Z += weight * rv.pdf(pos)

plt.figure(figsize=(8, 8))
plt.contourf(X, Y, Z, levels=20, cmap='plasma')
plt.colorbar()
plt.title('Initial 2D Gaussian Mixture')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('beamer_reports/mixture.png')
plt.close()

T = 8.0
n_steps = 100
dt = T/n_steps
t = np.linspace(0, T, n_steps)

trajectories = []
for mean in means:
    for _ in range(2):
        traj = np.zeros((n_steps, 2))
        traj[0] = np.random.multivariate_normal(mean, covs[0])
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt)/5, size=2)
            traj[i] = traj[i-1] + 3*dW
            
        trajectories.append(traj)

plt.figure(figsize=(8, 8))
for i, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], label=f'Trajectory {i+1}')
plt.title('Brownian Motion Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('beamer_reports/brownian.png')
plt.close()

final_trajectories = np.array([traj[-1] for traj in trajectories])
final_mean = np.mean(final_trajectories, axis=0)
final_cov = np.cov(final_trajectories.T)

Z_final = multivariate_normal(final_mean, final_cov).pdf(pos)

plt.figure(figsize=(8, 8))
plt.contourf(X, Y, Z_final, levels=20, cmap='plasma')
plt.colorbar()
plt.title('Final 2D Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('beamer_reports/gaussian.png')
plt.close()

with open('beamer_reports/brownian.md', 'w') as f:
    f.write('# Brownian Motion Trajectories\n\n')
    f.write('Initial positions:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'Trajectory {i+1}: ({traj[0,0]:.3f}, {traj[0,1]:.3f})\n')
    f.write('\nFinal positions:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'Trajectory {i+1}: ({traj[-1,0]:.3f}, {traj[-1,1]:.3f})\n')
    
    f.write('\nFull trajectories:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'\nTrajectory {i+1}:\n')
        for t in range(len(traj)):
            f.write(f't={t*dt:.2f}: ({traj[t,0]:.3f}, {traj[t,1]:.3f})\n')
