import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

cars_indices = [i for i, (_, label) in enumerate(trainset) if label == 1]
img1 = trainset[cars_indices[0]][0].to(device)
img2 = trainset[cars_indices[1]][0].to(device)

T = 400.0
n_steps = 100
times = np.logspace(0, np.log10(T), n_steps)
dt = np.diff(times, prepend=0)[1:]
sigma = 0.04 # find by hand 
theta = 0

trajectories = []
for start_img in [img1, img2]:
    traj = torch.zeros((n_steps,) + start_img.shape, device=device)
    traj[0] = start_img
    
    for i in range(1, n_steps):
        dW = torch.randn(start_img.shape, device=device) * np.sqrt(dt[i-1])
        drift = -theta * (traj[i-1] - start_img) * dt[i-1]
        diffusion = sigma * dW
        traj[i] = traj[i-1] + drift + diffusion
        
    trajectories.append(traj.cpu().numpy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Ornstein-Uhlenbeck Diffusion of CIFAR Images')

def animate(frame):
    ax1.clear()
    ax2.clear()
    
    ax1.imshow(np.transpose(trajectories[0][frame], (1, 2, 0)))
    ax2.imshow(np.transpose(trajectories[1][frame], (1, 2, 0)))
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.figtext(0.5, 0.02, f'Time: {times[frame]:.2f}s', ha='center')

anim = FuncAnimation(fig, animate, frames=n_steps, interval=50)
anim.save('results/cifar/cifar_diffusion.gif', writer='pillow')
plt.close()
