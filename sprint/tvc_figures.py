import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost in at least one dimension
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_pareto_frontier(task, layer, data, output_path):
    """
    Create a Pareto frontier plot for a specific task and layer
    """
    plt.figure(figsize=(10, 6))
    
    # Different markers for different techniques
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for technique, param_data in data.items():
        for param, value_data in param_data.items():
            l0_values = []
            loss_values = []
            
            # Extract L0 and loss values
            for value, metrics in value_data.items():
                l0_values.append(metrics[0])  # L0
                loss_values.append(metrics[1])  # Loss
            
            # Convert to numpy arrays for efficient computation
            points = np.column_stack((l0_values, loss_values))
            
            # Find Pareto frontier points
            pareto_mask = is_pareto_efficient(points)
            
            # Sort points for line plotting
            pareto_points = points[pareto_mask]
            sorted_indices = np.argsort(pareto_points[:, 0])
            pareto_points = pareto_points[sorted_indices]
            
            # Plot all points and Pareto frontier
            label = f"{technique} - {param}"
            marker = markers[hash(technique + param) % len(markers)]
            plt.scatter(l0_values, loss_values, alpha=0.5, marker=marker, label=label)
            plt.plot(pareto_points[:, 0], pareto_points[:, 1], '--', alpha=0.7)

    plt.xlabel('L0 Norm')
    plt.ylabel('Loss')
    plt.title(f'Pareto Frontier for {task} - Layer {layer}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save plot with tight layout to accommodate legend
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load JSON data
    with open('data/l1_sweep_results.json', 'r') as f:
        data = json.load(f)
    
    # Process each task:layer combination
    for key, value in data.items():
        task, layer = key.split(':')
        output_path = Path(f'data/tvc_tuning/{task}_{layer}.png')
        plot_pareto_frontier(task, layer, value, output_path)
        print(f"Generated plot for {task} - Layer {layer}")

if __name__ == "__main__":
    main()