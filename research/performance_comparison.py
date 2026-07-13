import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_top_features_performance(results_df, model_class, X_full_train, X_full_test, 
                                       y_train, y_test, 
                                       percentages=[1, 5, 10, 20, 50, 100], 
                                       ranking_col='mean_A',
                                       model_params=None,
                                       log_x=False):
    
    if model_params is None:
        model_params = {}
    
    # Get available features (intersection of results and actual data columns)
    available_features = [f for f in results_df['feature'].tolist() if f in X_full_train.columns]
    n_available = len(available_features)
    n_total = len(results_df)
    
    if n_available < n_total:
        print(f"⚠ Warning: Only {n_available}/{n_total} features from results found in data")
    
    if n_available == 0:
        raise ValueError("No matching features found between results_df and X_full_train!")
    
    # Sort available features by importance
    features_ranked = results_df[results_df['feature'].isin(available_features)].sort_values(
        ranking_col, ascending=False
    )['feature'].tolist()
    
    # Store results
    results = []

    model_params_base = dict(model_params)
    
    for pct in percentages:
        # Select top percentage of features
        n_features = max(1, int(len(features_ranked) * pct / 100))
        top_features = features_ranked[:n_features]
        
        # Filter data to use only top features
        X_train_subset = X_full_train[top_features]
        X_test_subset = X_full_test[top_features]

        model_params_local = dict(model_params_base)
        is_xgboost_model = getattr(model_class, "__module__", "").startswith("xgboost")
        has_categorical = isinstance(X_train_subset, pd.DataFrame) and any(
            pd.api.types.is_categorical_dtype(dtype) for dtype in X_train_subset.dtypes
        )
        if is_xgboost_model and has_categorical:
            model_params_local.setdefault("enable_categorical", True)
        
        # Train model
        model = model_class(**model_params_local)
        model.fit(X_train_subset, y_train)
        
        # Predict
        y_pred = model.predict(X_test_subset)
        y_pred_proba = model.predict_proba(X_test_subset)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # AUC from sklearn's roc_auc_score (binary or multiclass).
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            try:
                auc_macro = roc_auc_score(y_test, y_pred_proba[:, 1])
            except ValueError:
                auc_macro = np.nan
            auc_micro = auc_macro
        else:
            try:
                auc_micro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='micro')
            except ValueError:
                auc_micro = np.nan

            try:
                auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except ValueError:
                auc_macro = np.nan
        
        results.append({
            'Percentage': pct,
            'N_Features': n_features,
            'Accuracy': accuracy,
            'Precision (macro)': precision_macro,
            'F1 (macro)': f1_macro,
            'AUC Micro': auc_micro,
            'AUC Macro': auc_macro
        })
        
        print(f"✓ {pct}% ({n_features} features): Acc={accuracy:.4f}, F1={f1_macro:.4f}, AUC(macro)={auc_macro:.4f}")
    
    # Create results table
    results_table = pd.DataFrame(results)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy and F1
    ax1 = axes[0, 0]
    ax1.plot(results_table['Percentage'], results_table['Accuracy'], 
             marker='o', linewidth=2, markersize=8, label='Accuracy')
    ax1.plot(results_table['Percentage'], results_table['F1 (macro)'], 
             marker='s', linewidth=2, markersize=8, label='F1 (macro)')
    ax1.set_xlabel('Top Features (%)' + (' [log scale]' if log_x else ''), fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Accuracy and F1 Score vs Feature Percentage', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])
    if log_x:
        ax1.set_xscale('log')
    
    # Plot 2: Precision
    ax2 = axes[0, 1]
    ax2.plot(results_table['Percentage'], results_table['Precision (macro)'], 
             marker='^', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Top Features (%)' + (' [log scale]' if log_x else ''), fontsize=11)
    ax2.set_ylabel('Precision (macro)', fontsize=11)
    ax2.set_title('Precision vs Feature Percentage', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])
    if log_x:
        ax2.set_xscale('log')
    
    # Plot 3: AUC scores
    ax3 = axes[1, 0]
    ax3.plot(results_table['Percentage'], results_table['AUC Micro'], 
             marker='D', linewidth=2, markersize=8, label='AUC Micro', color='purple')
    ax3.plot(results_table['Percentage'], results_table['AUC Macro'], 
             marker='v', linewidth=2, markersize=8, label='AUC Macro', color='orange')
    ax3.set_xlabel('Top Features (%)' + (' [log scale]' if log_x else ''), fontsize=11)
    ax3.set_ylabel('AUC Score', fontsize=11)
    ax3.set_title('AUC Scores vs Feature Percentage', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1.05])
    if log_x:
        ax3.set_xscale('log')
    
    # Plot 4: All metrics combined
    ax4 = axes[1, 1]
    ax4.plot(results_table['Percentage'], results_table['Accuracy'], 
             marker='o', linewidth=2, label='Accuracy', alpha=0.7)
    ax4.plot(results_table['Percentage'], results_table['Precision (macro)'], 
             marker='^', linewidth=2, label='Precision', alpha=0.7)
    ax4.plot(results_table['Percentage'], results_table['F1 (macro)'], 
             marker='s', linewidth=2, label='F1', alpha=0.7)
    ax4.plot(results_table['Percentage'], results_table['AUC Macro'], 
             marker='D', linewidth=2, label='AUC Macro', alpha=0.7)
    ax4.set_xlabel('Top Features (%)' + (' [log scale]' if log_x else ''), fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('All Metrics Combined', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1.05])
    if log_x:
        ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return results_table, fig