import numpy as np
import matplotlib.pyplot as plt

class Msdnn2EnhancedVisualization():
    def __init__(self, subs_number, layer_sizes):
        self.subs_number = subs_number
        self.layer_sizes = layer_sizes
        self.node_positions = {}
        self.node_size = 3000 / max(layer_sizes) if max(layer_sizes) > 10 else 50
        self.calculate_positions()

    def calculate_positions(self):
        max_layer_size = max(self.layer_sizes)
        layer_height = 20 / max_layer_size
        scale_spacing = 80.0 / self.subs_number

        for sub in range(self.subs_number):
            for layer_idx, size in enumerate(self.layer_sizes):
                vertical_spacing = layer_height / max(size - 1, 1)
                positions = np.column_stack((
                    np.full(size, layer_idx + scale_spacing * sub),
                    np.linspace(-vertical_spacing * size / 2, vertical_spacing * size / 2, size)
                ))
                self.node_positions[f'scale{sub}_layer{layer_idx}'] = positions

    def draw(self):
        fig_width = 10 + self.subs_number * 2
        fig_height = 10 + max(self.layer_sizes) / 5
        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()

        # Draw connections
        for sub in range(self.subs_number):
            for layer_idx in range(len(self.layer_sizes) - 1):
                layer1_positions = self.node_positions[f'scale{sub}_layer{layer_idx}']
                layer2_positions = self.node_positions[f'scale{sub}_layer{layer_idx + 1}']
                self.draw_connections(layer1_positions, layer2_positions, ax)

        # Draw nodes
        for positions in self.node_positions.values():
            ax.scatter(positions[:, 0], positions[:, 1], s=self.node_size, color='black', zorder=4)

        ax.axis('off')
        plt.show()

    def draw_connections(self, layer1_positions, layer2_positions, ax):
        for pos1 in layer1_positions:
            for pos2 in layer2_positions:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='grey', linestyle='-', linewidth=0.5)

# Testing the enhanced visualization with a large middle layer
large_layer_sizes = [2, 10, 10, 3, 1]
ms_enhanced_viz = Msdnn2EnhancedVisualization(4, large_layer_sizes)
ms_enhanced_viz.draw()
