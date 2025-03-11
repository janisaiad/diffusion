from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

idx = (mnist.targets == 0) | (mnist.targets == 1)
mnist.data = mnist.data[idx]
mnist.targets = mnist.targets[idx]

n_samples = 1000
selected_idx = torch.randperm(len(mnist.data))[:n_samples]
X = mnist.data[selected_idx].float() / 255.0
labels = mnist.targets[selected_idx]
X = X.reshape(n_samples, -1)

pca = PCA(n_components=2)
X_pca = torch.tensor(pca.fit_transform(X))

T = 200
timesteps = torch.exp(torch.linspace(np.log(1e-3), np.log(8), T))

def compute_P_t_e(x, data, t):
    n = len(data)
    Delta_t = 1 - torch.exp(-2*t)
    diff = x.unsqueeze(0) - data * torch.exp(-t)
    P_t_e = torch.mean(torch.exp(-torch.sum(diff**2, dim=1)/(2*Delta_t))) / ((2*np.pi*Delta_t)**(x.shape[0]/2))
    return P_t_e

def compute_gaussian_entropy(t):
    d = X_pca.shape[1]
    return d/2 * (1 + np.log(2*np.pi))

def compute_entropy_and_excess(data, t, n_samples=10000):
    Delta_t = 1 - torch.exp(-2*t)
    samples = torch.randn(n_samples, data.shape[1]) * torch.sqrt(Delta_t) + \
             data[torch.randint(0, len(data), (n_samples,))] * torch.exp(-t)
    
    entropy = torch.tensor(0.0)
    for x in samples:
        p = compute_P_t_e(x, data, t)
        if p > 0:
            entropy -= (1/n_samples) * torch.log(p)
            
    gaussian_entropy = compute_gaussian_entropy(t)
    excess_entropy = entropy.item() - gaussian_entropy
    
    return entropy.item(), excess_entropy

entropies = []
excess_entropies = []
for t in tqdm(timesteps):
    entropy, excess = compute_entropy_and_excess(X_pca, t)
    entropies.append(entropy)
    excess_entropies.append(excess)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.semilogx(timesteps, entropies, 'k-', label='Total entropy s(t)', linewidth=2)
ax1.semilogx(timesteps, [compute_gaussian_entropy(t) for t in timesteps], 'k--', label='Gaussian entropy', linewidth=2)
ax1.set_xlabel('Time t (log scale)')
ax1.set_ylabel('Entropy')
ax1.set_title('Total and Gaussian entropy over time')
ax1.grid(True)
ax1.legend()

ax2.semilogx(timesteps, excess_entropies, 'k-', linewidth=2)
ax2.set_xlabel('Time t (log scale)')
ax2.set_ylabel('Excess entropy f(t)')
ax2.set_title('Excess entropy over time')
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/MNIST/entropy_analysis.png')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

scatter = ax1.scatter([], [], c=[], cmap=plt.cm.RdBu, alpha=0.5)
ax1.set_xlim(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1)
ax1.set_ylim(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1)

line, = ax2.semilogx([], [], 'k-', linewidth=2)
ax2.set_xlim(1e-3, 8)
ax2.set_ylim(min(excess_entropies), max(excess_entropies))
ax2.set_xlabel('Time t (log scale)')
ax2.set_ylabel('Excess entropy f(t)')
ax2.grid(True)

def update(frame):
    t = timesteps[frame]
    noise_scale = torch.sqrt(1 - torch.exp(-2*t))
    noisy_data = X_pca * torch.exp(-t) + noise_scale * torch.randn_like(X_pca)
    
    scatter.set_offsets(noisy_data.numpy())
    scatter.set_array(labels.numpy())
    ax1.set_title(f't = {t:.3e}')
    
    line.set_data(timesteps[:frame+1], excess_entropies[:frame+1])
    
    return scatter, line

anim = FuncAnimation(fig, update, frames=T, interval=50, blit=True)

anim.save('results/MNIST/diffusion_and_excess_entropy.gif', writer='pillow')
plt.close()
