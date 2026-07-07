import numpy as np 
import os
import matplotlib.pyplot as plt 

def plot_spatial_error_heatmap(coords, targets_np, preds_np, save_path=None, title="Spatial Error Heatmap (MAE)"):
    spot_errors = np.mean(np.abs(targets_np - preds_np), axis=1)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=spot_errors, cmap='coolwarm', s=15, alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Absolute Error (MAE)', rotation=270, labelpad=15)

    plt.title(title, fontsize=14)
    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial error heatmap saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_spatial_comparison_heatmaps(coords, targets_np, preds_np, mse_val, pcc_val, save_path=None, sample_id=""):
    true_expr = np.mean(targets_np, axis=1)
    pred_expr = np.mean(preds_np, axis=1)
    spot_errors = np.mean(np.abs(targets_np - preds_np), axis=1)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    vmin = min(np.min(true_expr), np.min(pred_expr))
    vmax = max(np.max(true_expr), np.max(pred_expr))

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    title_str = f"Spatial Expression Analysis - {sample_id}\n" if sample_id else "Spatial Expression Analysis\n"
    title_str += f"Mean Gene-wise PCC: {pcc_val:.4f}  |  Median L2 Error: {mse_val:.4f}"
    
    fig.suptitle(title_str, fontsize=18, fontweight='bold', y=1.08)
    sc1 = axes[0].scatter(x_coords, y_coords, c=true_expr, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax)
    axes[0].set_title("True Expression (Mean across genes)", fontsize=14)
    axes[0].axis('equal')
    cbar1 = fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Expression Level', rotation=270, labelpad=15)

    sc2 = axes[1].scatter(x_coords, y_coords, c=pred_expr, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted Expression (Mean across genes)", fontsize=14)
    axes[1].axis('equal')
    cbar2 = fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Expression Level', rotation=270, labelpad=15)
    sc3 = axes[2].scatter(x_coords, y_coords, c=spot_errors, cmap='coolwarm', s=15, alpha=0.8)
    axes[2].set_title("Spatial Error (MAE)", fontsize=14)
    axes[2].axis('equal')
    cbar3 = fig.colorbar(sc3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Mean Absolute Error', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        print(f"Comparison heatmaps saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()

