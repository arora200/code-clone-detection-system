import pandas as pd
import base64
import os

def generate_html_report():
    # Load results
    try:
        results_df = pd.read_csv("./results/comparison_results.csv")
    except FileNotFoundError:
        print("Error: comparison_results.csv not found.")
        return

    # --- Start HTML ---
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Clone Detection Analysis Report</title>
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 1em; }
            .container { max-width: 1200px; margin: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Code Clone Detection System: Analysis Report</h1>
            
            <h2>Executive Summary</h2>
            <p>
                This report summarizes the performance of various machine learning models for the task of code clone detection. 
                The analysis includes traditional machine learning models, a meta-classifier ensemble, and deep learning models.
                The key findings indicate that while traditional models perform exceptionally well on this dataset, deep learning models like CodeBERT show potential but require further tuning and more complex architectures for optimal performance.
            </p>

            <h2>Model Performance Comparison</h2>
            {table}

            <h2>Visualizations</h2>
    """

    # --- Embed Table ---
    html = html.replace("{table}", results_df.to_html(index=False))

    # --- Embed Standard Images ---
    viz_files = [
        "performance_metrics_comparison.png",
        "time_complexity_analysis.png",
        "performance_heatmap.png",
        "radar_chart_top5.png",
        "model_type_distributions.png",
        "learning_curves.png"
    ]

    for fname in viz_files:
        try:
            with open(os.path.join("./results", fname), "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
            html += f'<h3>{fname.replace("_", " ").replace(".png", "").title()}</h3>'
            html += f'<img src="data:image/png;base64,{encoded_string}" alt="{fname}">'
        except FileNotFoundError:
            html += f"<p><i>Visualization '{fname}' not found.</i></p>"

    # --- Embed Advanced Images ---
    html += "<h2>Advanced Visualizations</h2>"
    adv_viz_files = [
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png"
    ]

    for fname in adv_viz_files:
        try:
            with open(os.path.join("./results/advanced", fname), "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
            html += f'<h3>{fname.replace("_", " ").replace(".png", "").title()}</h3>'
            html += f'<img src="data:image/png;base64,{encoded_string}" alt="{fname}">'
        except FileNotFoundError:
            html += f"<p><i>Advanced visualization '{fname}' not found.</i></p>"


    # --- Final Experiment Section ---
    html += """
    <h2>Final Experiment: ExtraTrees as Meta-Classifier</h2>
    <p>
        Based on the strong performance of the single <code>ExtraTreesClassifier</code>, a final experiment was conducted to test if using it as a meta-classifier in a stacking ensemble would yield even better results. The goal was to find the most robust and performant model.
    </p>
    <h3>Experiment Results</h3>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>Model</th>
          <th>Accuracy</th>
          <th>F1 Score</th>
          <th>Training Time (ms)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Single ExtraTrees</td>
          <td>0.995</td>
          <td>0.994975</td>
          <td>90</td>
        </tr>
        <tr>
          <td>Stacking_Ensemble_ET_Meta</td>
          <td>0.995</td>
          <td>0.994975</td>
          <td>3075</td>
        </tr>
      </tbody>
    </table>
    <h3>Conclusion</h3>
    <p>
        The experiment demonstrated that while the stacking ensemble with an ExtraTrees meta-classifier performs exceptionally well, it offers no performance improvement over the single ExtraTrees model. However, it comes with a significantly higher computational cost, taking over 30 times longer to train.
    </p>
    <p>
        <b>Final Verdict:</b> The <b>Single ExtraTrees Classifier</b> is the most robust and practical solution, providing the best possible performance with the highest efficiency and lowest complexity.
    </p>
    """

    # --- End HTML ---
    html += """
        </div>
    </body>
    </html>
    """

    with open("analysis_report.html", "w") as f:
        f.write(html)

    print("HTML report generated successfully: analysis_report.html")

if __name__ == '__main__':
    generate_html_report()
