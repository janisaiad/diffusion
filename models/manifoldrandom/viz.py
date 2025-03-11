import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Callable

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

def exp(x: np.ndarray) -> np.ndarray:
    return np.exp(-x**2)

NONLINEARITIES = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'sin': sin,
    'exp': exp
}

def visualize_manifold_embedding(n_samples=1000, 
                               dim_intrinsic=2, 
                               dim_ambient=3, 
                               nonlinearity: Callable = tanh,
                               nonlinearity_name: str = 'tanh'):
    F = np.random.randn(dim_ambient, dim_intrinsic) / np.sqrt(dim_intrinsic)
    z = np.random.randn(n_samples, dim_intrinsic)
    X = nonlinearity(z @ F.T)
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=z[:, 0], cmap='viridis', alpha=0.6)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title("Vue 1 (45°)")
    
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=z[:, 0], cmap='viridis', alpha=0.6)
    ax2.view_init(elev=20, azim=120)
    ax2.set_title("Vue 2 (120°)")
    
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=z[:, 0], cmap='viridis', alpha=0.6)
    ax3.view_init(elev=20, azim=225)
    ax3.set_title("Vue 3 (225°)")
    
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(X[:, 0], X[:, 1], X[:, 2], c=z[:, 0], cmap='viridis', alpha=0.6)
    ax4.view_init(elev=90, azim=0)
    ax4.set_title("Vue 4 (dessus)")
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.grid(True)
    
    fig.suptitle(f"Manifold {dim_intrinsic}D dans R^{dim_ambient} avec {nonlinearity_name}\n"
                f"Coloration selon la première coordonnée latente", 
                fontsize=16, y=0.95)
    
    plt.colorbar(scatter1, ax=[ax1, ax2, ax3, ax4], label='Première coordonnée latente')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = Path(__file__).parent.parent.parent / "results" / "manifold_viz"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"manifold_D{dim_intrinsic}_N{dim_ambient}_{nonlinearity_name}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot sauvegardé dans: {save_path}")
    plt.close()

def main():
    n_samples = 3000
    
    for name, func in NONLINEARITIES.items():
        print(f"Génération du manifold avec {name}...")
        visualize_manifold_embedding(
            n_samples=n_samples,
            dim_intrinsic=2,
            dim_ambient=3,
            nonlinearity=func,
            nonlinearity_name=name
        )

if __name__ == "__main__":
    main()
