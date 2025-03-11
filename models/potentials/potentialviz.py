import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
import matplotlib
import tqdm
matplotlib.use('Agg')

def compute_potential_gpu(q: cp.ndarray, t: float, d: int, mu_tilde: float = 1.0) -> cp.ndarray:
    return 0.5 * q**2 - 2 * mu_tilde**2 * cp.log(cp.cosh(q * cp.exp(-t) * cp.sqrt(d)))

def calculate_optimal_q_range(t: float, d: int, mu_tilde: float = 1.0) -> tuple:
    scaling_factor = cp.exp(-t) * cp.sqrt(d)
    
    if scaling_factor > 1:
        q_scale = 2.5 / scaling_factor
        return (-q_scale, q_scale)
    else:
        return (-2.5, 2.5)

def create_potential_animation(d: int = 100, 
                             base_q_range: tuple = (-3, 3),
                             n_points: int = 1000,
                             n_frames: int = 2000,
                             fps: int = 30,
                             adaptive_scaling: bool = True) -> None:
    t_switch = 0.5 * cp.log(d)
    
    t_min, t_max = 0.01 * t_switch, 5 * t_switch
    times = cp.logspace(cp.log10(t_min), cp.log10(t_max), n_frames)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [])
    ax.grid(True)
    
    if not adaptive_scaling:
        q_range = base_q_range
        q = cp.linspace(q_range[0], q_range[1], n_points)
        
        ax.set_xlim(q_range)
        V_min = float(cp.min(compute_potential_gpu(q, times[-1], d)))
        V_max = float(cp.max(compute_potential_gpu(q, times[0], d)))
        ax.set_ylim(V_min - 0.5, V_max + 0.5)
    
    ax.set_xlabel('q')
    ax.set_ylabel('V(q,t)')
    title = ax.set_title('')
    
    minima_points = cp.array([-1.0, 1.0])
    minima_scatter = ax.scatter([], [], color='red', s=50, zorder=3)
    
    zero_line = ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    t_crit_text = ax.text(0.02, 0.98, f't_s ≈ {float(t_switch):.3f}', 
                         transform=ax.transAxes, fontsize=10,
                         verticalalignment='top')
    
    def init():
        line.set_data([], [])
        minima_scatter.set_offsets(np.empty((0, 2)))
        return line, minima_scatter
    
    def animate(frame):
        t = times[frame]
        
        if adaptive_scaling:
            q_range = calculate_optimal_q_range(t, d)
            q = cp.linspace(q_range[0], q_range[1], n_points)
            ax.set_xlim(q_range)
        else:
            q = cp.linspace(base_q_range[0], base_q_range[1], n_points)
        
        V = compute_potential_gpu(q, t, d)
        
        line.set_data(cp.asnumpy(q), cp.asnumpy(V))
        
        if adaptive_scaling:
            V_min = float(cp.min(V))
            V_max = float(cp.max(V))
            
            y_margin = 0.1 * (V_max - V_min) if V_max > V_min else 0.5
            ax.set_ylim(V_min - y_margin, V_max + y_margin)
        
        ratio = float(t/t_switch)
        title.set_text(f't/tS = {ratio:.2f}')
        
        if t < t_switch:
            minima_y = compute_potential_gpu(minima_points, t, d)
            minima_data = np.column_stack((cp.asnumpy(minima_points), cp.asnumpy(minima_y)))
            minima_scatter.set_offsets(minima_data)
            alpha_value = float(min(1.0, (t_switch-t)/(0.5*t_switch)))
            minima_scatter.set_alpha(alpha_value)
        else:
            minima_scatter.set_offsets(np.empty((0, 2)))
            
        return line, minima_scatter, title
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_frames, interval=1000//fps, 
                                 blit=True)
    
    save_dir = Path('results/potentials')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    scaling_type = "adaptive" if adaptive_scaling else "fixed"
    anim.save(save_dir / f'potential_evolution_d{d}_{scaling_type}.gif', 
              writer='pillow', fps=fps)
    plt.close()

    cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    for d in tqdm.tqdm(range(10, 1000, 10)):
        create_potential_animation(d=d, adaptive_scaling=True)
        print(f"Animation créée pour d={d} avec mise à l'échelle adaptative")