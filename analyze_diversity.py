import pandas as pd
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PREDICTIONS_DIR = 'D:/Task/Model_Predictions'

REPORTS_DIR = 'D:/Task/Trained_Models'
ANALYSIS_OUTPUT_DIR = 'D:/Task/Analysis_Results'

# --- SCRIPT ---

def analyze_model_diversity():
    """
    Loads prediction CSVs, analyzes model performance and diversity,
    and generates correlation/disagreement matrices.
    """
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    
    # Find all prediction files
    try:
        pred_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.endswith('_predictions.csv')]
        if not pred_files:
            print(f"❌ No prediction files found in '{PREDICTIONS_DIR}'.")
            print("   Please run 'train_and_evaluate.py' for each model first.")
            return
    except FileNotFoundError as e:
        print(f"❌ Error: The directory '{PREDICTIONS_DIR}' was not found. Details: {e}")
        print("   Please ensure you have run the training script and the directory was created correctly.")
        return
    
    model_names = sorted([f.replace('_predictions.csv', '') for f in pred_files])
    print(f"Found predictions for models: {model_names}")
    
    # --- 1. Load all predictions into a dictionary ---
    predictions = {}
    for model_name in model_names:
        file_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv")
        df = pd.read_csv(file_path)
        # Sort by filename to ensure all dataframes are aligned for comparison
        predictions[model_name] = df.sort_values(by='filename').reset_index(drop=True)
    
    # --- 2. Create a summary of individual model performance (F1-Score) ---
    performance_summary = {}
    for model_name in model_names:
        report_path = os.path.join(REPORTS_DIR, f'{model_name}_classification_report.json')
        try:
            # The report is a JSON file, which pandas can read directly
            report_df = pd.read_json(report_path)
            # Get the weighted average F1-score
            f1_score = report_df.loc['f1-score', 'weighted avg']
            performance_summary[model_name] = f1_score
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠️ Warning: Report file not found or missing 'weighted avg' for {model_name}. Skipping performance summary.")
            performance_summary[model_name] = 'N/A'
    
    print("\n--- Individual Model Performance (Weighted F1-Score) ---")
    # Sort by score, handling 'N/A' values
    sorted_perf = sorted(performance_summary.items(), key=lambda item: item[1] if isinstance(item[1], float) else -1, reverse=True)
    for model, score in sorted_perf:
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  - {model:<20}: {score_str}")
    
    # --- 3. Prediction Correlation Analysis ---
    # We will correlate the error vectors. An error is 1 if prediction is wrong, 0 if correct.
    error_df = pd.DataFrame()
    # Use the first model's dataframe to get the true labels
    first_model_name = model_names[0]
    error_df['true_label'] = predictions[first_model_name]['true_label']

    for name, df in predictions.items():
        error_df[name] = (df['true_label'] != df['predicted_label']).astype(int)

    error_correlation = error_df.drop(columns=['true_label']).corr()

    # --- Plotting the heatmap using only Matplotlib ---
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(error_correlation, cmap='coolwarm', vmin=-1, vmax=1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(range(len(error_correlation.columns)))
    ax.set_yticks(range(len(error_correlation.columns)))
    ax.set_xticklabels(error_correlation.columns)
    ax.set_yticklabels(error_correlation.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(error_correlation.index)):
        for j in range(len(error_correlation.columns)):
            # Use a different text color for dark vs. light cells for readability
            color = "white" if abs(error_correlation.iloc[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{error_correlation.iloc[i, j]:.2f}", ha="center", va="center", color=color)

    ax.set_title("Pairwise Error Correlation Matrix", fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'error_correlation_matrix.png')
    plt.savefig(save_path)
    print(f"\n✅ Error correlation matrix saved to: {save_path}")
    plt.show()
    
    print("\n--- Interpretation ---")
    print("Look for models with HIGH F1-scores and LOW correlation values in the matrix.")
    print("A low correlation (e.g., < 0.5) means the models make different mistakes, which is ideal for an ensemble.")

if __name__ == '__main__':
    analyze_model_diversity()