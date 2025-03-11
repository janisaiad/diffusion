import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from tqdm import tqdm

class ExcessEntropyAnalysis:
    def __init__(self, alpha: float = 1.0, dim: int = 2):
        self.d = dim
        self.alpha = alpha
        self.n = int(cp.round(cp.exp(alpha * dim)))
        print(f"Pour α = {alpha:.2f}, d = {dim}: utilisation de n = {self.n} points")
        self.initial_points = cp.random.randn(self.n, dim)
        self.initial_points = self.initial_points / cp.linalg.norm(self.initial_points, axis=1)[:, cp.newaxis]
    
    def compute_f_t(self, t: float) -> float:
        delta_t = 1 - cp.exp(-2 * t)
        s_sep = self.alpha + 0.5 * (1 + cp.log(2 * cp.pi * delta_t))
        means = self.initial_points * cp.exp(-t)
        diff = means[:, cp.newaxis, :] - means[cp.newaxis, :, :]
        quad_terms = cp.sum(diff**2, axis=2)
        log_terms = -0.5 * quad_terms / (2 * delta_t) - \
                   (self.d/2) * cp.log(4 * cp.pi * delta_t) - \
                   cp.log(self.n)
        
        s_emp = -cp.mean(cp.log(cp.mean(cp.exp(log_terms), axis=1))) / self.d
        return float(cp.asnumpy(s_sep - s_emp))

    def analyze_regimes(self, t_range=(0.01, 5), n_points=1000):
        times = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_points)
        f_t = np.array([self.compute_f_t(t) for t in tqdm(times, desc="Calcul f(t)")])
        collapse_idx = np.argmin(np.abs(f_t))
        t_c = times[collapse_idx]
        
        f_inf = self.alpha
        return times, f_t, t_c, f_inf

def plot_regimes_analysis():
    alphas = np.linspace(0.1, 3.5, 30)
    
    plt.figure(figsize=(15, 10))
    t_cs = []
    
    for alpha in alphas:
        print(f"\nAnalyse pour α = {alpha:.2f}")
        analyzer = ExcessEntropyAnalysis(alpha=alpha)
        times, f_t, t_c, f_inf = analyzer.analyze_regimes()
        t_cs.append(t_c)
        
        plt.plot(times, f_t, '-', label=f'α = {alpha:.1f}')
        plt.axvline(x=t_c, linestyle=':', alpha=0.3, color='g')
        plt.axhline(y=f_inf, linestyle='--', alpha=0.3, color='r')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.xlabel('Temps t (échelle log)')
    plt.ylabel('Excès d\'entropie f(t)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.text(0.02, 2, 'Régime III\n(Mémorisation)', ha='left', va='top')
    plt.text(0.5, 2, 'Régime II\n(Généralisation)', ha='center', va='top')
    plt.text(3, 2, 'Régime I\n(Bruit)', ha='right', va='top')
    
    plt.title('Analyse des régimes de la dynamique rétrograde\n' + 
              'f(t) = s_sep(t) - s(t) pour différentes valeurs de α')
    
    save_dir = Path(__file__).parent.parent.parent / "results" / "entropy_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"regimes_analysis_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, t_cs, 'bo-')
    plt.xlabel('α = log(n)/d')
    plt.ylabel('Temps de collapse t_C')
    plt.grid(True)
    plt.title('Temps de collapse en fonction de α\n' +
              'Plus α est grand, plus le régime III est réduit')
    
    save_path = save_dir / f"tc_vs_alpha_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nRésultats :")
    for alpha, t_c in zip(alphas, t_cs):
        print(f"α = {alpha:.2f}:")
        print(f"  t_C = {t_c:.3f}")
        print(f"  f(t>>1) = {alpha:.2f}")

if __name__ == "__main__":
    plot_regimes_analysis()
