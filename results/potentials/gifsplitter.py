import imageio
import numpy as np


gif_path = 'results/potentials/potential_evolution_d500_adaptive.gif'
output_path = 'results/potentials/potential_evolution_d500_adaptive_1060_1500.gif'


with imageio.get_reader(gif_path) as reader:
    frames = [frame for frame in reader]
selected_frames = frames[1060:1500]
with imageio.get_writer(output_path, mode='I') as writer:
    for frame in selected_frames:
        writer.append_data(frame)
