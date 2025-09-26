"""
Part 6: Visualization Functions
"""

from utils import *

def create_visualizations(results_df, save_path="./results"):
    """Create comprehensive visualizations for the comparison"""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc']
    titles = ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'ROC AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        sorted_df = results_df.sort_values(metric, ascending=False).head(10)
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
        
        # Color bars by model type
        colors = {'Traditional ML': 'blue', 'Meta-Classifier': 'red', 'Deep Learning': 'green'}
        for i, model_type in enumerate(sorted_df['type']):
            bars[i].set_color(colors.get(model_type, 'gray'))
        
        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        ax.set_title(f'Top 10 Models by {title}')
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time Complexity Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by time
    time_df = results_df.sort_values('time_ms').head(15)
    
    bars = ax.bar(range(len(time_df)), time_df['time_ms'])
    
    # Color by type
    for i, model_type in enumerate(time_df['type']):
        bars[i].set_color(colors.get(model_type, 'gray'))
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Training Time Comparison (Top 15 Fastest Models)')
    ax.set_xticks(range(len(time_df)))
    ax.set_xticklabels(time_df['model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/time_complexity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of Model Performance
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for heatmap
    heatmap_data = results_df.set_index('model')[metrics].head(15)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, ax=ax)
    ax.set_title('Performance Metrics Heatmap (Top 15 Models)')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Radar Chart for Top 5 Models
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    top_5 = results_df.nlargest(5, 'f1_score')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for idx, row in top_5.iterrows():
        values = [row[m] for m in metrics]
        values += [values[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'][:30])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.upper() for t in titles])
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Multi-Metric Comparison', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/radar_chart_top5.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Box Plot for Model Types
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for box plot
        data_by_type = []
        labels = []
        for model_type in results_df['type'].unique():
            data_by_type.append(results_df[results_df['type'] == model_type][metric].values)
            labels.append(model_type)
        
        bp = ax.boxplot(data_by_type, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], ['blue', 'red', 'green'][:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution by Model Type')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/model_type_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Learning Curves Simulation (for demonstration)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simulate learning curves for top models
    top_models = results_df.nlargest(5, 'f1_score')
    
    epochs = np.arange(1, 51)
    for idx, row in top_models.iterrows():
        # Simulate a learning curve
        final_score = row['f1_score']
        curve = final_score * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 0.01, len(epochs))
        ax.plot(epochs, curve, label=row['model'][:30], linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Simulated Learning Curves - Top 5 Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations saved to {save_path}/")

if __name__ == '__main__':
    print("="*60)
    print("VISUALIZATION")
    print("="*60)

    # Load SOTA comparison results
    try:
        results_df = pd.read_json('./data/sota_results.json')
    except FileNotFoundError:
        print("Error: sota_results.json not found. Please run Phase 5 (SOTA Comparison) first.")
        exit()
    except Exception as e:
        print(f"Error loading SOTA results: {e}")
        exit()

    create_visualizations(results_df)
    print("Visualizations generated successfully.")
