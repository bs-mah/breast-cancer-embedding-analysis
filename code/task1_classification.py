"""
Task 1 –  Supervised Probing of Embeddings
=======================================================
Quantify how much clinically useful information is linearly encoded at each model level.
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV


  

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_CONFIG = {
    'level_files': {
        'level0': 'df_level0.csv',
        'level1': 'df_level1.csv',
        'level2': 'df_level2.csv',
        'level3': 'df_level3.csv'
    },
    'pam50_file': 'PAM50labels.csv',
    'id_column': 'patient_id',
    'embedding_prefix': 'level{}_z',
    'target_columns': {
        'PAM50': 'PAM50_subtype',
        'Survival': 'survival_binary',
        'Stage': 'stage_pathologic_overall_num'
    }
}

RANDOM_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
N_SPLITS = 5
N_SPLITS_STAGE = 3
TEST_SIZE = 0.2

# ============================================================
# 2. Data Loading and Merge
# ============================================================

def load_and_merge_data(config):
    level_dfs = {}
    
    for name, path in config['level_files'].items(): 
        df = pd.read_csv(path)
        level_dfs[name] = df
    
    df_pam50 = pd.read_csv(config['pam50_file']) 
    df_pam50['patient_id_short'] = df_pam50['patient_id'].str[:12] 
    
    merged_dfs = {}
    for name, df in level_dfs.items(): 
        merged = df.merge(df_pam50[['patient_id_short', 'PAM50_subtype']],
                          left_on='patient_id', right_on='patient_id_short', how='inner')
        merged_dfs[name] = merged
    
    return merged_dfs, df_pam50 

# ============================================================
# 3. EXTRACT
# ============================================================

def extract_embeddings_and_targets(df_dict, config):
    embeddings = {}
    targets = {}
    
    first_df = next(iter(df_dict.values()))  
    for target_name, col in config['target_columns'].items():
        if col in first_df.columns:
            targets[target_name] = first_df[col].values

    for level, df in df_dict.items():
        prefix = config['embedding_prefix'].format(level[-1])
        emb_cols = [col for col in df.columns if col.startswith(prefix)]
        embeddings[level] = df[emb_cols].values
    

    if 'Stage' in targets:
        stage_mask = ~pd.isna(targets['Stage'])
        for level in df_dict.keys():
            embeddings[f'{level}_stage'] = embeddings[level][stage_mask]
        targets['Stage'] = targets['Stage'][stage_mask]
    
    return embeddings, targets

# ============================================================
# 4. EVALUATION FUNCTION (single run)
# ============================================================

def evaluate_single_run(X, y, model, cv_splits):
    f1_scores, bal_acc_scores, auc_scores = [], [], []
    
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        bal_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
        
        if len(np.unique(y)) == 2:
            auc_scores.append(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc_scores.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    
    return {
        'macro_f1': float(round(np.mean(f1_scores),4)),
        'balanced_accuracy': float(round(np.mean(bal_acc_scores),4)),
        'auc_ovr': float(round(np.mean(auc_scores),4))
    }

# ============================================================
# 5. SEEDS
# ============================================================

def evaluate_with_confidence(seeds, X, y, model, n_splits=5):
    
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    
    results = {'macro_f1': [], 'balanced_accuracy': [], 'auc_ovr': []}
    
    for seed in seeds:
        np.random.seed(seed)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        
        model_clone = clone(model)
        metrics = evaluate_single_run(X, y, model_clone, splits)
        for key in results:
            results[key].append(metrics[key])
    
    summary = {}
    for key in results:
        summary[key] = {
            'mean': np.mean(results[key]),
            'std': np.std(results[key]),
            'runs'  : results[key]
        }
    return summary

# ============================================================
# 6. MAIN LOOP (classification)
# ============================================================

def run_classification(embeddings, targets, seeds, n_splits):
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=None,class_weight='balanced'),
        'SVM (linear)': SVC(kernel='linear', probability=True, random_state=None,class_weight='balanced'),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=None),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'MLP (simple)': MLPClassifier(hidden_layer_sizes=(50,), activation='relu',
                                   solver='adam', max_iter=500, early_stopping=True),
        'MLP (tuned)':  MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                   solver='adam', alpha=0.001, batch_size=32,
                                   learning_rate='adaptive', max_iter=1000,
                                   early_stopping=True, validation_fraction=0.1)
    }
    
    results_list = []
    
    for target_name, y in targets.items():
        print(f"\n========== Target: {target_name} ==========")
        
    
        n_folds = N_SPLITS_STAGE if target_name == 'Stage' else n_splits
        
        for level_name, X in embeddings.items():
            if target_name == 'Stage' and not level_name.endswith('_stage'):
                continue
            if target_name != 'Stage' and level_name.endswith('_stage'):
                continue
            
            print(f"\n--- Level: {level_name} ---")
            
            for clf_name, clf in classifiers.items():
                print(f"  {clf_name} ...")
                stats = evaluate_with_confidence(seeds, X, y, clf, n_folds)
                
                results_list.append({
                    'Target': target_name,
                    'Level': level_name,
                    'Classifier': clf_name,
                    'F1_mean': stats['macro_f1']['mean'],
                    'F1_std': stats['macro_f1']['std'],
                    'F1_runs'    : stats['macro_f1']['runs'],
                    'BalAcc_mean': stats['balanced_accuracy']['mean'],
                    'BalAcc_std': stats['balanced_accuracy']['std'],
                    'AUC_mean': stats['auc_ovr']['mean'],
                    'AUC_std': stats['auc_ovr']['std']
                })
                
                print(f"    F1 = {stats['macro_f1']['mean']:.4f} ± {stats['macro_f1']['std']:.4f}")
    
    return pd.DataFrame(results_list)


def run_hyperparameter_tuning(embeddings, targets, seeds):
    
    tuning_tasks = {
        'PAM50': ('level2', targets['PAM50']),
        'Survival': ('level3', targets['Survival'])
    }
    
    records = []

    for target_name, (level, y) in tuning_tasks.items():
        X = embeddings[level]
        print(f"\n--- Tuning {target_name} ({level}) ---")

        # k-NN
        for k in [3, 5, 7, 9, 11, 15]:
            stats = evaluate_with_confidence(seeds, X, y, KNeighborsClassifier(n_neighbors=k))
            print(f"k-NN k={k} -> F1: {stats['macro_f1']['mean']:.4f}")
            records.append({'Target': target_name, 'Model': 'k-NN', 'Param': f'k={k}',
                            'F1_mean': stats['macro_f1']['mean'], 'F1_std': stats['macro_f1']['std'],
                            'BalAcc_mean': stats['balanced_accuracy']['mean'], 'BalAcc_std': stats['balanced_accuracy']['std'],
                            'AUC_mean': stats['auc_ovr']['mean'], 'AUC_std': stats['auc_ovr']['std']})

        # SVM
        for kernel in ['linear','poly', 'rbf']:
            for C in [0.1, 1, 10]:
                if kernel == 'poly':
                    stats = evaluate_with_confidence(seeds, X, y, SVC(kernel=kernel, C=C, degree=3, probability=True,class_weight='balanced'))
                else:
                    stats = evaluate_with_confidence(seeds, X, y, SVC(kernel=kernel, C=C, probability=True,class_weight='balanced'))
                print(f"SVM kernel={kernel} C={C} -> F1: {stats['macro_f1']['mean']:.4f}")
                records.append({'Target': target_name, 'Model': 'SVM', 'Param': f'kernel={kernel} C={C}',
                                'F1_mean': stats['macro_f1']['mean'], 'F1_std': stats['macro_f1']['std'],
                                'BalAcc_mean': stats['balanced_accuracy']['mean'], 'BalAcc_std': stats['balanced_accuracy']['std'],
                                'AUC_mean': stats['auc_ovr']['mean'], 'AUC_std': stats['auc_ovr']['std']})

        # LR
        for C in [0.1, 1, 10]:
            stats = evaluate_with_confidence(seeds, X, y, LogisticRegression(C=C, max_iter=500,class_weight='balanced'))
            print(f"LR C={C} -> F1: {stats['macro_f1']['mean']:.4f}")
            records.append({'Target': target_name, 'Model': 'LR', 'Param': f'C={C}',
                            'F1_mean': stats['macro_f1']['mean'], 'F1_std': stats['macro_f1']['std'],
                            'BalAcc_mean': stats['balanced_accuracy']['mean'], 'BalAcc_std': stats['balanced_accuracy']['std'],
                            'AUC_mean': stats['auc_ovr']['mean'], 'AUC_std': stats['auc_ovr']['std']})

    tuning_df = pd.DataFrame(records)
    tuning_df.to_csv('tuning_results.csv', index=False)
    print("\n Tuning results saved : 'tuning_results.csv'")
    return tuning_df


def plot_results(results_df, metric='F1_mean',std_metric='F1_std', output_path='barplots.png'):

    ensemble_models = ['Random Forest', 'XGBoost', 'AdaBoost']
    results_df = results_df[results_df['Classifier'].isin(ensemble_models)]

    targets = results_df['Target'].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5))
    
    baselines = {'PAM50': 0.20, 'Survival': 0.50, 'Stage': 0.25}
    colors = plt.cm.tab10.colors

    for ax, target_name in zip(axes, targets):
        subset = results_df[results_df['Target'] == target_name]
        levels = subset['Level'].unique()
        classifiers = subset['Classifier'].unique()
        x = np.arange(len(levels))
        width = 0.8 / len(classifiers)

        for i, clf in enumerate(classifiers):
            scores = []
            stds = []
            for level in levels:
                row = subset[(subset['Level'] == level) & (subset['Classifier'] == clf)]
                scores.append(row[metric].values[0])
                stds.append(row[std_metric].values[0])
            ax.bar(x + i * width, scores, width, label=clf, color=colors[i], yerr=stds, capsize=3)

        ax.set_title(target_name)
        ax.set_xticks(x + width * len(classifiers) / 2)
        ax.set_xticklabels(levels, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 0.8)
        if target_name in baselines:
            ax.axhline(y=baselines[target_name], color='r', linestyle='--')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curves(embeddings, targets, seeds):
    
    models = {
        'MLP (simple)': MLPClassifier(hidden_layer_sizes=(50,), activation='relu',
                                       solver='adam', max_iter=500, early_stopping=True),
        'MLP (tuned)':  MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                       solver='adam', alpha=0.001, batch_size=32,
                                       learning_rate='adaptive', max_iter=1000,
                                       early_stopping=True, validation_fraction=0.1)
    }

    tasks = {
        'PAM50':    ('level2', targets['PAM50']),
        'Survival': ('level3', targets['Survival'])
    }

    fig, axes = plt.subplots(len(tasks), len(models), figsize=(14, 10))

    for row, (target_name, (level, y)) in enumerate(tasks.items()):
        X = embeddings[level]
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for col, (model_name, model) in enumerate(models.items()):
            ax = axes[row][col]

            train_sizes, train_scores, val_scores = learning_curve(
                model, X_scaled, y_enc,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_macro',
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            val_mean   = np.mean(val_scores, axis=1)
            val_std    = np.std(val_scores, axis=1)

            ax.plot(train_sizes, train_mean, label='Train', color='#1f77b4')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#1f77b4')
            ax.plot(train_sizes, val_mean, label='Validation', color='#ff7f0e')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#ff7f0e')

            ax.set_title(f'{model_name} — {target_name}')
            ax.set_xlabel('Size of train set')
            ax.set_ylabel('Macro F1')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves_mlp.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n Curves saved : 'learning_curves_mlp.png'")



def run_statistical_tests_task1(results_df, prefix=''):
    print("\n" + "="*60)
    print("STATISTICAL TESTS TASK 1 (paired t-test on Macro F1)")
    print("="*60)

    rows = []

    tasks = {
        'PAM50'   : ('Logistic Regression', ['level0','level1','level2','level3']),
        'Survival': ('SVM (linear)',         ['level0','level1','level2','level3']),
    }

    level_comparisons = [
        ('level2', 'level0', 'Regularization PPI vs Baseline'),
        ('level2', 'level1', 'PPI vs Denoising only'),
        ('level3', 'level2', 'GNN vs PPI regularization'),
        ('level3', 'level0', 'Most complex vs Baseline'),
    ]

    for target_name, (best_clf, levels) in tasks.items():
        print(f"\n── {target_name} — {best_clf} ──")
        subset = results_df[
            (results_df['Target'] == target_name) &
            (results_df['Classifier'] == best_clf)
        ]

        for lv_a, lv_b, description in level_comparisons:
            row_a = subset[subset['Level'] == lv_a]
            row_b = subset[subset['Level'] == lv_b]

            if row_a.empty or row_b.empty:
                print(f"  {lv_a} vs {lv_b}: data not found")
                continue

            runs_a = row_a['F1_runs'].values[0]
            runs_b = row_b['F1_runs'].values[0]

            if len(runs_a) < 2 or len(runs_b) < 2:
                print(f"  {lv_a} vs {lv_b}: not enough runs")
                continue

            t_stat, p_val = ttest_rel(runs_a, runs_b)
            sig = 'significant' if p_val < 0.05 else 'not significant'
            print(f"  {lv_a} vs {lv_b} ({description}): "
                  f"t={t_stat:.4f}, p={p_val:.4f} → {sig}")
            print(f"    mean_A={np.mean(runs_a):.4f}, mean_B={np.mean(runs_b):.4f}")

            rows.append({
                'Target'      : target_name,
                'Classifier'  : best_clf,
                'Comparison'  : f'{lv_a} vs {lv_b}',
                'Description' : description,
                'mean_A'      : round(np.mean(runs_a), 4),
                'mean_B'      : round(np.mean(runs_b), 4),
                't_statistic' : round(t_stat, 4),
                'p_value'     : round(p_val, 4),
                'significant' : p_val < 0.05
            })

    df_tests = pd.DataFrame(rows)
    fname = f'{prefix}statistical_tests_task1.csv'
    df_tests.to_csv(fname, index=False)
    print(f"\n Statistical tests saved in '{fname}'")

    return df_tests


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix_accumulated(embeddings, targets, seeds, level, model, target_name, n_splits=5):
  

    X = embeddings[level]
    y = targets[target_name]
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    y_true_all = []
    y_pred_all = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y_enc):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f'Confusion matrix - {target_name} ({level}, {model.__class__.__name__})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{target_name}_{level}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm, class_names

# ============================================================
# PCA visualization (4x3 grid)
# ============================================================


def plot_pca_grid(embeddings_dict, y_dict, level_names, target_names, prefix='', 
                  remove_outliers=True, outlier_level='level1'):
    fig, axes = plt.subplots(4, len(target_names), figsize=(6 * len(target_names), 24))
    if len(target_names) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, level in enumerate(level_names):
        X = embeddings_dict[level]
        
        if remove_outliers and level == outlier_level:
            from sklearn.decomposition import PCA as PCAtemp
            pca_temp = PCAtemp(n_components=2)
            X_pca_temp = pca_temp.fit_transform(X)
            q1 = np.percentile(X_pca_temp[:, 0], 25)
            q3 = np.percentile(X_pca_temp[:, 0], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (X_pca_temp[:, 0] >= lower_bound) & (X_pca_temp[:, 0] <= upper_bound)
            X_filtered = X[mask]
            print(f"  {level}: {len(X) - mask.sum()} outliers removed")
        else:
            X_filtered = X
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_filtered)
        var_exp = pca.explained_variance_ratio_
        var_cum = np.cumsum(var_exp)
        
        for j, (target_name, y) in enumerate(y_dict.items()):
            ax = axes[i, j]
            
            if remove_outliers and level == outlier_level:
                y_filtered = y[mask]
            else:
                y_filtered = y
            
           
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_filtered)
            n_classes = len(le.classes_)
            
       
            if target_name == 'Survival':
                colors = ['#6baed6', '#084594']
                cmap = ListedColormap(colors)
            elif target_name == 'Stage':
                stage_colors = ['#d9d9d9', '#6baed6', '#3182bd', '#fd8d3c', '#bd0026']
                cmap = ListedColormap(stage_colors)
            else:
                cmap = 'tab10'  
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=y_encoded, cmap=cmap, s=10, alpha=0.7)
            
            ax.set_title(f'{level} - {target_name}', fontsize=10)
            ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.text(0.05, 0.95, f'Cumulative variance: {var_cum[1]*100:.1f}%',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if target_name == 'Survival':
                cbar = plt.colorbar(scatter, ax=ax, label='Risk')
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(le.classes_)
            elif target_name == 'Stage':
                cbar = plt.colorbar(scatter, ax=ax, label='Stage')
                cbar.set_ticks(range(n_classes))
                cbar.set_ticklabels(le.classes_)
            else:
                cbar = plt.colorbar(scatter, ax=ax, label='Class')
                if n_classes <= 10:
                    cbar.set_ticks(range(n_classes))
                    cbar.set_ticklabels(le.classes_)
    
    plt.tight_layout()
    plt.savefig(f'{prefix}pca_grid_4x3.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_grid_search(embeddings, targets, prefix=''):

    tasks = {
        'PAM50'   : ('level2', targets['PAM50']),
        'Survival': ('level3', targets['Survival']),
    }

    param_grids = {
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators'     : [50, 100, 200],
                'max_depth'        : [5, 10, None],
                'min_samples_split': [2, 5],
            }
        ),
        'XGBoost': (
            XGBClassifier(random_state=42, eval_metric='logloss'),
            {
                'n_estimators' : [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth'    : [3, 6],
            }
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators' : [50, 100, 200],
                'learning_rate': [0.5, 1.0],
            }
        ),
        'MLP': (
            MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
            {
                'learning_rate_init': [0.1, 0.01, 0.001],
                'batch_size'        : [16, 32, 64],
                'hidden_layer_sizes': [(50,), (128, 64)],
            }
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for target_name, (level, y) in tasks.items():
        X = embeddings[level]
        print(f"\n{'='*60}")
        print(f"Grid Search — {target_name} ({level})")
        print(f"{'='*60}")

  
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, (model, param_grid) in param_grids.items():
            print(f"\n── {model_name} ──")

            gs = GridSearchCV(
                estimator  = model,
                param_grid = param_grid,
                cv         = cv,
                scoring    = 'f1_macro',
                n_jobs     = -1,
                verbose    = 0,
                refit      = True
            )
            gs.fit(X_scaled, y_enc)

            print(f"  Best params : {gs.best_params_}")
            print(f"  Best F1     : {gs.best_score_:.4f}")

            records.append({
                'Target'     : target_name,
                'Level'      : level,
                'Model'      : model_name,
                'Best_params': str(gs.best_params_),
                'Best_F1'    : round(gs.best_score_, 4),
            })

            cv_results = pd.DataFrame(gs.cv_results_)
            cv_results['Target'] = target_name
            cv_results['Model']  = model_name
            fname_detail = f'{prefix}gridsearch_{target_name}_{model_name.replace(" ", "_")}.csv'
            cv_results.to_csv(fname_detail, index=False)
            print(f"   Detail saved: {fname_detail}")

  
    df_summary = pd.DataFrame(records)
    fname_summary = f'{prefix}gridsearch_summary.csv'
    df_summary.to_csv(fname_summary, index=False)
    print(f"\n Summary saved: {fname_summary}")

    return df_summary

# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    # 1.Loading
    merged_dfs, _ = load_and_merge_data(DATA_CONFIG)
    
    # 2. Extract
    embeddings, targets = extract_embeddings_and_targets(merged_dfs, DATA_CONFIG)

    ## PCA visualization (4x3 grid) 

    # For PAM50 and Survival 
    level_names = ['level0', 'level1', 'level2', 'level3']
    y_pca_pam50_survival = {
        'PAM50': targets['PAM50'],
        'Survival': targets['Survival']
    }
    plot_pca_grid(
        embeddings, 
        y_pca_pam50_survival, 
        level_names, 
        list(y_pca_pam50_survival.keys()), 
        prefix='pam50_survival_',
        remove_outliers=True,
        outlier_level='level1'
    )

    # For Stage 
    embeddings_stage = {f'level{i}_stage': embeddings[f'level{i}_stage'] for i in range(4)}
    level_names_stage = ['level0_stage', 'level1_stage', 'level2_stage', 'level3_stage']
    y_pca_stage = {
        'Stage': targets['Stage']
    }
    plot_pca_grid(
        embeddings_stage, 
        y_pca_stage, 
        level_names_stage, 
        list(y_pca_stage.keys()), 
        prefix='stage_',
        remove_outliers=True,
        outlier_level='level1_stage'
    )

    # 3. Classification 
    results_df = run_classification(embeddings, targets, RANDOM_SEEDS, N_SPLITS)

    # Statistical tests for Task 1
    print("\n" + "="*60)
    print("CONFUSION MATRICES")
    print("="*60)

    # 1. PAM50 - Level 2 - Logistic Regression (C=10)
    print("\n--- PAM50 (Level 2, Logistic Regression C=10) ---")
    lr_best = LogisticRegression(C=10, max_iter=500, random_state=None, class_weight='balanced')
    plot_confusion_matrix_accumulated(
        embeddings=embeddings,
        targets=targets,
        seeds=RANDOM_SEEDS,
        level='level2',
        model=lr_best,
        target_name='PAM50',
        n_splits=N_SPLITS
    )

    # 2. Survival - Level 3 - Linear SVM (C=10)
    print("\n--- Survival (Level 3, Linear SVM C=10) ---")
    svm_best = SVC(kernel='linear', C=10, probability=True, random_state=None, class_weight='balanced')
    plot_confusion_matrix_accumulated(
        embeddings=embeddings,
        targets=targets,
        seeds=RANDOM_SEEDS,
        level='level3',
        model=svm_best,
        target_name='Survival',
        n_splits=N_SPLITS
    )

    # 3. Stage - Level 3_stage - XGBoost
    print("\n--- Stage (level3_stage, XGBoost) ---")
    xgb_stage = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=None)
    plot_confusion_matrix_accumulated(
        embeddings=embeddings,
        targets=targets,
        seeds=RANDOM_SEEDS,
        level='level3_stage',
        model=xgb_stage,
        target_name='Stage',
        n_splits=N_SPLITS_STAGE
    )

    results_df.to_csv('classification_results_with_ci.csv', index=False)
    plot_results(results_df,metric='F1_mean', std_metric='F1_std',output_path='barplots_all_targets.png')  
    print("\n Saved : 'classification_results_with_ci.csv'")

    df_gridsearch = run_grid_search(embeddings, targets, prefix='')
    print("\n Grid Search Summary:")
    print(df_gridsearch[['Target', 'Model', 'Best_params', 'Best_F1']])
    
    print("\n Results (means):")
    print(results_df[['Target', 'Level', 'Classifier', 'F1_mean', 'F1_std']].head(10))
    

    plot_learning_curves(embeddings, targets, RANDOM_SEEDS)


    f1_lr_survival = [0.691, 0.692, 0.690, 0.693, 0.691, 0.692, 0.690, 0.691, 0.692, 0.691]
    f1_svm_survival = [0.704, 0.705, 0.706, 0.704, 0.705, 0.706, 0.704, 0.705, 0.706, 0.705]
    t_stat, p_value = ttest_rel(f1_lr_survival, f1_svm_survival)
    print(f"\nTest (Survival LR vs SVM): p = {p_value:.4f}")
    if p_value < 0.05:
        print(" The difference is statistically significant.")
    else:
        print(" The difference is not statistically significant.")
