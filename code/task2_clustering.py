"""
Task 2 – Clustering Analysis of Embedding Space
=======================================================
Evaluate whether patient embeddings naturally organise into
clinically coherent groups without supervision.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_files': {
        'level0': 'df_level0.csv',
        'level1': 'df_level1.csv',
        'level2': 'df_level2.csv',
        'level3': 'df_level3.csv',
    },
    'pam50_file': 'PAM50labels.csv',

    'id_column': 'patient_id',
    'embedding_prefix': 'level{}_z',
    'pam50_subtype_col': 'PAM50_subtype',
    'survival_col': 'survival_binary',
    'stage_col': 'stage_pathologic_overall_num',
    'extra_clinical_cols' : ['T_stage', 'N_stage', 'M_stage'],
    

    # Clustering
    'k_range'             : range(2, 9),
    'k_fixed'             : 5,     

    # Algorithms to run 
    'algos'               : ['kmeans', 'agg', 'spectral', 'gmm', 'hdbscan'],

    # Hyperparameters
    'kmeans_n_init'       : 10,
    'hdbscan_min_cluster' : 5,
    'hdbscan_min_samples' : 1,
    'spectral_n_neighbors': 10,
    'spectral_n_components': 20,
    'gmm_n_init'          : 5,

    'output_prefix'       : '', 
}

RANDOM_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Colour / label maps for plots
ALGO_COLORS = {
    'kmeans'  : '#4C9BE8',
    'agg'     : '#E8834C',
    'spectral': '#6DBE6D',
    'gmm'     : '#C46DE8',
    'hdbscan' : '#E8C44C',
}
ALGO_LABELS = {
    'kmeans'  : 'K-Means',
    'agg'     : 'Agglomerative',
    'spectral': 'Spectral',
    'gmm'     : 'GMM',
    'hdbscan' : 'HDBSCAN',
}

# =============================================================================
# STEP 1 — Data Loading and Preparation
# =============================================================================

def load_data(config):
    dfs = {}
    for level, path in config['data_files'].items():
        dfs[level] = pd.read_csv(path)

    df_pam50 = pd.read_csv(config['pam50_file'])
    df_pam50['patient_id_short'] = df_pam50['patient_id'].str[:12]

    merged_dfs = {}
    for name, df in dfs.items():
        merged = df.merge(df_pam50[['patient_id_short', config['pam50_subtype_col']]],
                          left_on=config['id_column'], right_on='patient_id_short', how='inner')
        merged_dfs[name] = merged
    
    return merged_dfs


def extract_arrays(dfs, config):
    first_df  = next(iter(dfs.values()))
    pam50_col = config['pam50_subtype_col']
    surv_col  = config['survival_col']
    stage_col = config['stage_col']
 
    y_pam50    = first_df[pam50_col].values
    y_survival = first_df[surv_col].values
 
    print("\nMissing values (level0):")
    print(first_df[[pam50_col, surv_col, stage_col]].isnull().sum())
 
    X_levels = {}
    for level, df in dfs.items():
        prefix   = config['embedding_prefix'].format(level[-1])
        emb_cols = [c for c in df.columns if c.startswith(prefix)]
        X_levels[level] = df[emb_cols].values
 
    stage_mask = ~first_df[stage_col].isna()
    y_stage    = first_df.loc[stage_mask, stage_col].values
 
    X_stage = {}
    for level, df in dfs.items():
        prefix   = config['embedding_prefix'].format(level[-1])
        emb_cols = [c for c in df.columns if c.startswith(prefix)]
        X_stage[f'{level}_stage'] = df.loc[stage_mask, emb_cols].values
 
    # Stage binary
    y_stage_binary_full = np.full(len(first_df), np.nan)
    y_stage_binary_full[stage_mask] = np.where(y_stage <= 2, 0, 1)


    extra = {col: first_df[col].values
             for col in config['extra_clinical_cols']
             if col in first_df.columns}
    clinical_vars = {
        'PAM50'   : y_pam50,
        'Survival': y_survival,
        'Stage_binary' : y_stage_binary_full,
        **extra
    }
 
    return X_levels, X_stage, y_pam50, y_survival, y_stage, y_stage_binary_full, clinical_vars

# =============================================================================
# Step 2 — Normalization
# =============================================================================

def normalize(X_levels, X_stage):
    levels       = {name: StandardScaler().fit_transform(X) for name, X in X_levels.items()}
    levels_stage = {name: StandardScaler().fit_transform(X) for name, X in X_stage.items()}
    return levels, levels_stage


# =============================================================================
# Step 3 — Elbow Method
# =============================================================================

def plot_elbow(levels, k_range, title, filename):
    inertias = {name: [] for name in levels}
    for name, X in levels.items():
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias[name].append(km.inertia_)

    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4))
    if len(levels) == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, inertias.items()):
        ax.plot(list(k_range), vals, marker='o')
        ax.set_title(f'Elbow – {name}')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia (WCSS)')
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# Step 4 — Clustering
# =============================================================================

def run_clustering(levels, k_range, config,seeds):
    algos   = config['algos']
    results = {}

    if 'kmeans' in algos:
        print("\nK-Means...")
        results['kmeans'] = {}
        for name, X in levels.items():
            results['kmeans'][name] = {}
            for k in k_range:
                sil_runs, db_runs, labels_runs = [], [],[]
                for seed in seeds:
                    km     = KMeans(n_clusters=k, random_state=seed, n_init=config['kmeans_n_init'])
                    labels = km.fit_predict(X)
                    sil_runs.append(silhouette_score(X, labels))
                    db_runs.append(davies_bouldin_score(X, labels))
                    labels_runs.append(labels)
                results['kmeans'][name][k] = {
                    'labels'  : labels_runs[0],
                    'all_labels': labels_runs,
                    'sil'     : np.mean(sil_runs),
                    'sil_std' : np.std(sil_runs),
                    'sil_runs'  : sil_runs,
                    'db'      : np.mean(db_runs),
                    'db_std'  : np.std(db_runs)
                }
            print(f"  {name} ✓")

    if 'agg' in algos:
        print("\nAgglomerative...")
        results['agg'] = {}
        for name, X in levels.items():
            results['agg'][name] = {}
            for k in k_range:
                agg    = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = agg.fit_predict(X)
                results['agg'][name][k] = {
                    'labels': labels,
                    'sil'   : silhouette_score(X, labels),
                    'db'    : davies_bouldin_score(X, labels)
                }
            print(f"  {name} ✓")

    if 'hdbscan' in algos:
        print("\nHDBSCAN...")
        results['hdbscan'] = {}
        for name, X in levels.items():
            clusterer  = HDBSCAN(min_cluster_size=config['hdbscan_min_cluster'],
                                  min_samples=config['hdbscan_min_samples'])
            labels     = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(sum(labels == -1))
            mask       = labels != -1
            sil = silhouette_score(X[mask], labels[mask]) if n_clusters >= 2 else None
            db  = davies_bouldin_score(X[mask], labels[mask]) if n_clusters >= 2 else None
            results['hdbscan'][name] = {
                'labels': labels, 
                'sil': sil, 
                'db': db,
                'n_clusters': n_clusters, 
                'noise': n_noise
            }
            print(f"  {name} → clusters: {n_clusters}, noise: {n_noise} ({n_noise/len(X)*100:.1f}%)")

            if name == 'level3' :
                cluster_sizes = pd.Series(labels).value_counts().sort_index()

                print(f"For level 3 : {cluster_sizes}")
                
    if 'spectral' in algos:
        print("\nSpectral Clustering (10 runs)...")
        results['spectral'] = {}
        for name, X in levels.items():
            results['spectral'][name] = {}
            for k in k_range:
                print(f"  {name}, k={k}...", end=' ', flush=True)
                sil_runs, db_runs, labels_runs = [], [], []
                labels_final = None
                for seed in seeds:
                    spec = SpectralClustering(
                        n_clusters=k, affinity='nearest_neighbors',
                        n_neighbors=config['spectral_n_neighbors'],
                        n_components=config['spectral_n_components'],
                        random_state=seed, n_jobs=-1
                    )
                    labels = spec.fit_predict(X)
                    sil_runs.append(silhouette_score(X, labels))
                    db_runs.append(davies_bouldin_score(X, labels))
                    labels_runs.append(labels)
                    if seed == seeds[0]:
                        labels_final = labels
                results['spectral'][name][k] = {
                    'labels': labels_final,
                    'all_labels': labels_runs,
                    'sil': np.mean(sil_runs),
                    'sil_std': np.std(sil_runs),
                    'sil_runs'  : sil_runs,
                    'db': np.mean(db_runs),
                    'db_std': np.std(db_runs)
                }
                print("✓")

    if 'gmm' in algos:
        print("\nGMM...")
        results['gmm'] = {}
        for name, X in levels.items():
            results['gmm'][name] = {}
            for k in k_range:
                sil_runs, db_runs, bic_runs, aic_runs, labels_runs = [], [], [], [], []
                for seed in seeds:
                    gmm    = GaussianMixture(n_components=k, random_state=seed, n_init=config['gmm_n_init'])
                    labels = gmm.fit_predict(X)
                    sil_runs.append(silhouette_score(X, labels))
                    db_runs.append(davies_bouldin_score(X, labels))
                    bic_runs.append(gmm.bic(X))
                    aic_runs.append(gmm.aic(X))
                    labels_runs.append(labels)
                gmm_final = GaussianMixture(n_components=k, random_state=seeds[0], n_init=config['gmm_n_init'])
                labels_final = gmm_final.fit_predict(X)
                results['gmm'][name][k] = {
                    'labels'  : labels_final,
                    'all_labels': labels_runs,
                    'sil'     : np.mean(sil_runs),
                    'sil_std' : np.std(sil_runs),
                    'sil_runs'  : sil_runs,
                    'db'      : np.mean(db_runs),
                    'db_std'  : np.std(db_runs),
                    'bic'     : np.mean(bic_runs),
                    'aic'     : np.mean(aic_runs)
                }
            print(f"  {name} ✓")

    return results


# =============================================================================
# Step 5 — Metrics
# =============================================================================

def print_metrics(results, levels, k_range):
    algos_with_k = [a for a in results if a != 'hdbscan']
 
    print("\n" + "="*60)
    print("SILHOUETTE SCORES (mean ± std)")
    print("="*60)
    for algo in algos_with_k:
        print(f"\n── {algo.upper()} ──")
        for name in levels:
            print(f"  {name}:")
            for k in k_range:
                sil = results[algo][name][k]['sil']
                std = results[algo][name][k].get('sil_std', 0.0)
                print(f"    k={k}: {sil:.4f} ± {std:.4f}")
 
    if 'hdbscan' in results:
        print("\n── HDBSCAN ──")
        for name in levels:
            sil = results['hdbscan'][name]['sil']
            print(f"  {name}: {round(sil, 4) if sil is not None else 'N/A'}")
 
    print("\n" + "="*60)
    print("DAVIES-BOULDIN INDEX (mean ± std)")
    print("="*60)
    for algo in algos_with_k:
        print(f"\n── {algo.upper()} ──")
        for name in levels:
            print(f"  {name}:")
            for k in k_range:
                db  = results[algo][name][k]['db']
                std = results[algo][name][k].get('db_std', 0.0)
                print(f"    k={k}: {db:.4f} ± {std:.4f}")
 
    if 'hdbscan' in results:
        print("\n── HDBSCAN ──")
        for name in levels:
            db = results['hdbscan'][name]['db']
            print(f"  {name}: {round(db, 4) if db is not None else 'N/A'}")
 

# =============================================================================
# Step 6 — ARI / NMI
# =============================================================================

def print_ari_nmi(results, levels, y_pam50, k_fixed):
    print("\n" + "="*60)
    print(f"ARI AND NMI vs PAM50 (k={k_fixed})")
    print("="*60)
    for algo, data in results.items():
        print(f"\n── {algo.upper()} ──")
        for name in levels:
            entry = data[name][k_fixed] if algo != 'hdbscan' else data[name]
            all_labels = entry.get('all_labels', [entry['labels']]) 
            aris = [adjusted_rand_score(y_pam50, lbl) for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_pam50, lbl) for lbl in all_labels]
            print(f"  {name} → ARI: {np.mean(aris):.4f} ± {np.std(aris):.4f}, "
                  f"NMI: {np.mean(nmis):.4f} ± {np.std(nmis):.4f}")


def print_ari_nmi_stage(results, levels, y_stage_binary, k_fixed):
    print("\n" + "="*60)
    print(f"ARI AND NMI vs STAGE BINARY (k={k_fixed})")
    print("="*60)
    mask = ~np.isnan(y_stage_binary)
    for algo, data in results.items():
        print(f"\n── {algo.upper()} ──")
        for name in levels:
            entry = data[name][k_fixed] if algo != 'hdbscan' else data[name]
            all_labels = entry.get('all_labels', [entry['labels']])
            aris = [adjusted_rand_score(y_stage_binary[mask], lbl[mask]) 
                    for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_stage_binary[mask], lbl[mask]) 
                    for lbl in all_labels]
            print(f"  {name} → ARI: {np.mean(aris):.4f} ± {np.std(aris):.4f}, "
                  f"NMI: {np.mean(nmis):.4f} ± {np.std(nmis):.4f}")
            

# =============================================================================
# Step 7 — Chi-square
# =============================================================================

def print_chi2(results, results_stage, levels, levels_stage, clinical_vars, y_stage, k_fixed):
    print("\n" + "="*60)
    print(f"CHI-SQUARE TESTS (k={k_fixed})")
    print("="*60)
 
    for algo, data in results.items():
        print(f"\n── {algo.upper()} – PAM50/Survival/Clinical ──")
        for name in levels:
            labels = data[name][k_fixed]['labels'] if algo != 'hdbscan' else data[name]['labels']
            for var_name, y_var in clinical_vars.items():
                mask  = ~pd.isnull(y_var)
                table = pd.crosstab(labels[mask], y_var[mask])
                chi2, p, dof, _ = chi2_contingency(table)
                print(f"  {name}, {var_name}: p={p:.4f} → {'significant' if p < 0.05 else 'not significant'}")
 
        print(f"\n── {algo.upper()} – Stage ──")
        stage_data = results_stage.get(algo, {})
        for name in levels_stage:
            if name not in stage_data:
                continue
            labels = stage_data[name][k_fixed]['labels'] if algo != 'hdbscan' else stage_data[name]['labels']
            table  = pd.crosstab(labels, y_stage)
            chi2, p, dof, _ = chi2_contingency(table)
            print(f"  {name}, Stage: p={p:.4f} → {'significant' if p < 0.05 else 'not significant'}")
 

# =============================================================================
# Step 8 — Visualizations
# =============================================================================

def plot_bic_aic(results, levels, k_range, prefix=''):
    if 'gmm' not in results:
        return
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4))
    if len(levels) == 1:
        axes = [axes]
    for ax, name in zip(axes, levels):
        bics = [results['gmm'][name][k]['bic'] for k in k_range]
        aics = [results['gmm'][name][k]['aic'] for k in k_range]
        ax.plot(list(k_range), bics, marker='o', label='BIC')
        ax.plot(list(k_range), aics, marker='s', label='AIC')

        best_k = list(k_range)[np.argmin(bics)]
        ax.axvline(best_k, color='#4C9BE8', linestyle=':', alpha=0.6)
        ax.text(best_k + 0.1, max(bics) * 0.98, f'k={best_k}', color='#4C9BE8', fontsize=8)

        ax.set_title(f'GMM BIC/AIC – {name}')
        ax.set_xlabel('k')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.suptitle('GMM BIC/AIC', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{prefix}gmm_bic_aic.png', dpi=300, bbox_inches='tight')
    plt.show()
 
def plot_pca_clusters(results, levels, level_name='level3', k_fixed=5, prefix='', y_pam50=None):
    X     = levels[level_name]
    X_pca = PCA(n_components=2).fit_transform(X)
 
    algos_to_plot = {}
    for algo, data in results.items():
        if level_name not in data:
            continue
        labels = data[level_name][k_fixed]['labels'] if algo != 'hdbscan' else data[level_name]['labels']
        algos_to_plot[algo] = labels
 
    n_cols = len(algos_to_plot) + (1 if y_pam50 is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
 
    for ax, (algo, labels) in zip(axes, algos_to_plot.items()):
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                        cmap='tab10', s=12, alpha=0.7, linewidths=0)
        ax.set_title(f'{ALGO_LABELS.get(algo, algo)}', fontsize=10, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        plt.colorbar(sc, ax=ax, label='Cluster', shrink=0.8)
        ax.grid(True, linestyle='--', alpha=0.2)
 
    if y_pam50 is not None:
        ax = axes[-1]
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y_pam50)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_enc,
                        cmap='tab10', s=12, alpha=0.7, linewidths=0)
        ax.set_title('PAM50 (reference)', fontsize=10, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_ticks(range(len(le.classes_)))
        cbar.set_ticklabels(le.classes_)
        ax.grid(True, linestyle='--', alpha=0.2)
 
    fig.suptitle(f'PCA 2D — {level_name} (k={k_fixed})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'{prefix}clusters_pca_{level_name}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {fname}")
 
def plot_kmeans_levels_comparison(results, levels, k_fixed=5, prefix=''):
    if 'kmeans' not in results:
        return
    level_names = list(levels.keys())
    fig, axes   = plt.subplots(1, len(level_names), figsize=(5 * len(level_names), 5))
    if len(level_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, level_names):
        X      = levels[name]
        labels = results['kmeans'][name][k_fixed]['labels']
        X_pca  = PCA(n_components=2).fit_transform(X)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
        ax.set_title(f'K-Means (k={k_fixed}) – {name}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(f'{prefix}clusters_kmeans_comparison.png', dpi=300)
    plt.show()

def plot_silhouette_db_barplot(results, levels, k_fixed, prefix=''):
    algo_names  = list(results.keys())
    level_names = list(levels.keys())
    n_algos     = len(algo_names)
    width       = 0.8 / n_algos
    x           = np.arange(len(level_names))
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    for metric_idx, (metric_key, std_key, ax, title, better) in enumerate([
        ('sil', 'sil_std', axes[0], 'Silhouette Score  (↑ better)', 'high'),
        ('db',  'db_std',  axes[1], 'Davies-Bouldin Index  (↓ better)', 'low'),
    ]):
        for i, algo in enumerate(algo_names):
            means, stds = [], []
            for name in level_names:
                if algo == 'hdbscan':
                    val = results[algo][name].get(metric_key)
                    means.append(val if val is not None else 0)
                    stds.append(0)
                else:
                    entry = results[algo][name][k_fixed]
                    means.append(entry.get(metric_key, 0))
                    stds.append(entry.get(std_key, 0))
 
            bars = ax.bar(
                x + i * width, means, width,
                yerr=stds, capsize=3,
                label=ALGO_LABELS.get(algo, algo),
                color=ALGO_COLORS.get(algo, '#aaaaaa'),
                alpha=0.85, error_kw={'linewidth': 1.2}
            )
 
        ax.set_xticks(x + width * (n_algos - 1) / 2)
        ax.set_xticklabels(level_names, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Level', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
 
    fig.suptitle(f'Clustering quality metrics (k={k_fixed})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'{prefix}barplot_sil_db.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {fname}")
 
def plot_ari_nmi_heatmap(results, levels, y_pam50, k_fixed, prefix=''):
 
    algo_names  = list(results.keys())
    level_names = list(levels.keys())
 
    ari_matrix = np.zeros((len(algo_names), len(level_names)))
    nmi_matrix = np.zeros((len(algo_names), len(level_names)))
 
    for i, algo in enumerate(algo_names):
        for j, name in enumerate(level_names):
            entry = results[algo][name][k_fixed] if algo != 'hdbscan' else results[algo][name]
            all_labels = entry.get('all_labels', [entry['labels']])
            aris = [adjusted_rand_score(y_pam50, lbl) for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_pam50, lbl) for lbl in all_labels]
            ari_matrix[i, j] = np.mean(aris)
            nmi_matrix[i, j] = np.mean(nmis)
 
    cmap = LinearSegmentedColormap.from_list('clin', ['#f7fbff', '#2171b5'])
    ylabels = [ALGO_LABELS.get(a, a) for a in algo_names]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    for ax, matrix, title in [
        (axes[0], ari_matrix, 'ARI vs PAM50'),
        (axes[1], nmi_matrix, 'NMI vs PAM50'),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0)
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels(level_names, fontsize=9)
        ax.set_yticks(range(len(algo_names)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)

        for i in range(len(algo_names)):
            for j in range(len(level_names)):
                val = matrix[i, j]
                color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
 
    fig.suptitle(f'Clinical coherence — ARI & NMI vs PAM50 subtypes (k={k_fixed})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'{prefix}heatmap_ari_nmi.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {fname}")
 
def plot_confusion_clusters_pam50(results, levels, y_pam50, k_fixed, prefix='', level_name=None):
 
    algo_names = list(results.keys())
    le = LabelEncoder()
    pam50_enc  = le.fit_transform(y_pam50)
    pam50_labels = le.classes_
 
    n = len(algo_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
 
    cmap = LinearSegmentedColormap.from_list('conf', ['#ffffff', '#08519c'])
 
    for ax, algo in zip(axes, algo_names):
        if level_name is not None:
            best_level = level_name
        else:
            best_level = max(
                levels.keys(),
                key=lambda name: adjusted_rand_score(
                    y_pam50,
                    results[algo][name][k_fixed]['labels'] if algo != 'hdbscan'
                    else results[algo][name]['labels']
                )
            )
 
        entry  = results[algo][best_level][k_fixed] if algo != 'hdbscan' else results[algo][best_level]
        labels = entry['labels']
 
        ct = pd.crosstab(labels, y_pam50, normalize='index')
        ct = ct.reindex(columns=pam50_labels, fill_value=0)
 
        im = ax.imshow(ct.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(pam50_labels)))
        ax.set_xticklabels(pam50_labels, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels([f'C{c}' for c in ct.index], fontsize=8)
        ax.set_xlabel('PAM50 subtype', fontsize=8)
        ax.set_ylabel('Cluster', fontsize=8)
        ax.set_title(f'{ALGO_LABELS.get(algo, algo)}\n({best_level})', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Proportion')
 
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.values[i, j]
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
 
    fig.suptitle(f'Cluster composition vs PAM50 subtypes (k={k_fixed})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'{prefix}confusion_clusters_pam50.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {fname}")
 


def save_results_to_csv(results, results_stage, levels, levels_stage, k_range, filename='clustering_results.csv'):
    rows = []
    
    # PAM50 / Survival
    for algo, data in results.items():
        for level_name in levels:
            if algo == 'hdbscan':
                rows.append({
                    'Target': 'PAM50/Survival',
                    'Algorithm': algo.upper(),
                    'Level': level_name,
                    'k': 'auto',
                    'Silhouette': data[level_name]['sil'],
                    'Silhouette_std': 0,
                    'Davies_Bouldin': data[level_name]['db'],
                    'Davies_Bouldin_std': 0,
                    'Clusters': data[level_name]['n_clusters'],
                    'Noise': data[level_name]['noise']
                })
            else:
                for k in k_range:
                    rows.append({
                        'Target': 'PAM50/Survival',
                        'Algorithm': algo.upper(),
                        'Level': level_name,
                        'k': k,
                        'Silhouette': data[level_name][k]['sil'],
                        'Silhouette_std': data[level_name][k].get('sil_std', 0),
                        'Davies_Bouldin': data[level_name][k]['db'],
                        'Davies_Bouldin_std': data[level_name][k].get('db_std', 0),
                    })
    
    # Stage 
    for algo, data in results_stage.items():
        for level_name in levels_stage:
            if algo == 'hdbscan':
                rows.append({
                    'Target': 'Stage',
                    'Algorithm': algo.upper(),
                    'Level': level_name,
                    'k': 'auto',
                    'Silhouette': data[level_name]['sil'],
                    'Silhouette_std': 0,
                    'Davies_Bouldin': data[level_name]['db'],
                    'Davies_Bouldin_std': 0,
                    'Clusters': data[level_name]['n_clusters'],
                    'Noise': data[level_name]['noise']
                })
            else:
                for k in k_range:
                    rows.append({
                        'Target': 'Stage',
                        'Algorithm': algo.upper(),
                        'Level': level_name,
                        'k': k,
                        'Silhouette': data[level_name][k]['sil'],
                        'Silhouette_std': data[level_name][k].get('sil_std', 0),
                        'Davies_Bouldin': data[level_name][k]['db'],
                        'Davies_Bouldin_std': data[level_name][k].get('db_std', 0),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

def save_ari_nmi_to_csv(results, levels, y_pam50, k_fixed, filename='ari_nmi_results.csv'):
    rows = []
    for algo, data in results.items():
        for name in levels:
            entry = data[name][k_fixed] if algo != 'hdbscan' else data[name]
            all_labels = entry.get('all_labels', [entry['labels']])
            aris = [adjusted_rand_score(y_pam50, lbl) for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_pam50, lbl) for lbl in all_labels]
            rows.append({
                'Algorithm'  : algo.upper(),
                'Level'      : name,
                'k'          : k_fixed if algo != 'hdbscan' else entry.get('n_clusters', 'auto'),
                'ARI_mean'   : round(np.mean(aris), 4),
                'ARI_std'    : round(np.std(aris), 4),
                'NMI_mean'   : round(np.mean(nmis), 4),
                'NMI_std'    : round(np.std(nmis), 4),
                'n_runs'     : len(all_labels)
            })
    pd.DataFrame(rows).to_csv(filename, index=False)

def save_ari_nmi_stage_to_csv(results, levels, y_stage_binary, k_fixed, 
                               filename='ari_nmi_stage_binary_results.csv'):
    rows = []
    mask = ~np.isnan(y_stage_binary)
    for algo, data in results.items():
        for name in levels:
            entry = data[name][k_fixed] if algo != 'hdbscan' else data[name]
            all_labels = entry.get('all_labels', [entry['labels']])
            aris = [adjusted_rand_score(y_stage_binary[mask], lbl[mask]) 
                    for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_stage_binary[mask], lbl[mask]) 
                    for lbl in all_labels]
            rows.append({
                'Algorithm' : algo.upper(),
                'Level'     : name,
                'k'         : k_fixed if algo != 'hdbscan' else entry.get('n_clusters', 'auto'),
                'ARI_mean'  : round(np.mean(aris), 4),
                'ARI_std'   : round(np.std(aris), 4),
                'NMI_mean'  : round(np.mean(nmis), 4),
                'NMI_std'   : round(np.std(nmis), 4),
                'n_runs'    : len(all_labels)
            })
    pd.DataFrame(rows).to_csv(filename, index=False)


# =============================================================================
# Tuning HDBSCAN
# =============================================================================

def tune_hdbscan(levels, y_pam50, level_name='level3',
                 min_cluster_sizes=None, min_samples_range=None, prefix=''):
    if min_cluster_sizes is None:
        min_cluster_sizes = [5, 10, 20, 30, 50, 75, 100]
    if min_samples_range is None:
        min_samples_range = [1, 5, 10, 20, 30]
 
    X = levels[level_name]
 
    n_rows = len(min_cluster_sizes)
    n_cols = len(min_samples_range)
 
    mat_clusters = np.zeros((n_rows, n_cols))
    mat_noise    = np.zeros((n_rows, n_cols))
    mat_sil      = np.full((n_rows, n_cols), np.nan)
    mat_ari      = np.full((n_rows, n_cols), np.nan)
    mat_nmi      = np.full((n_rows, n_cols), np.nan)
 
    rows = []  
 
    print(f"\n── HDBSCAN tuning on {level_name} ──")
    print(f"\n  {'min_cluster_size':<20} {'min_samples':<15} {'n_clusters':<12} "
          f"{'noise %':<12} {'Silhouette':<12} {'ARI':<10} {'NMI':<10}")
    print(f"  {'-'*90}")
 
    for i, mcs in enumerate(min_cluster_sizes):
        for j, ms in enumerate(min_samples_range):
            hdb    = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = hdb.fit_predict(X)
 
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(sum(labels == -1))
            noise_pct  = n_noise / len(X) * 100
 
            mat_clusters[i, j] = n_clusters
            mat_noise[i, j]    = noise_pct
 
            mask = labels != -1
            if n_clusters >= 2 and mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                ari = adjusted_rand_score(y_pam50[mask], labels[mask])
                nmi = normalized_mutual_info_score(y_pam50[mask], labels[mask])
            else:
                sil, ari, nmi = np.nan, np.nan, np.nan
 
            mat_sil[i, j] = sil
            mat_ari[i, j] = ari
            mat_nmi[i, j] = nmi
 
            sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
            ari_str = f"{ari:.4f}" if not np.isnan(ari) else "N/A"
            nmi_str = f"{nmi:.4f}" if not np.isnan(nmi) else "N/A"
 
            print(f"  {mcs:<20} {ms:<15} {n_clusters:<12} "
                  f"{noise_pct:<12.1f} {sil_str:<12} {ari_str:<10} {nmi_str:<10}")
 
            rows.append({
                'min_cluster_size': mcs,
                'min_samples'     : ms,
                'n_clusters'      : n_clusters,
                'noise_pct'       : round(noise_pct, 2),
                'silhouette'      : round(sil, 4) if not np.isnan(sil) else None,
                'ari'             : round(ari, 4) if not np.isnan(ari) else None,
                'nmi'             : round(nmi, 4) if not np.isnan(nmi) else None,
            })
 

    df_tuning = pd.DataFrame(rows)
    fname_csv = f'{prefix}hdbscan_tuning_{level_name}.csv'
    df_tuning.to_csv(fname_csv, index=False)
 
    # ── Heatmaps ──
    cmap_clusters = LinearSegmentedColormap.from_list('cl', ['#f7fbff', '#08519c'])
    cmap_noise    = LinearSegmentedColormap.from_list('ns', ['#f7fcf5', '#08519c'])
    cmap_sil      = LinearSegmentedColormap.from_list('sl', ['#fff5f0', '#08519c'])
 
    xlabels = [str(ms)  for ms  in min_samples_range]
    ylabels = [str(mcs) for mcs in min_cluster_sizes]
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    for ax, matrix, cmap, title, fmt in [
        (axes[0], mat_clusters, cmap_clusters, 'Number of clusters',      '{:.0f}'),
        (axes[1], mat_noise,    cmap_noise,    '% noise',                 '{:.1f}%'),
        (axes[2], mat_sil,      cmap_sil,      'Silhouette Score',        '{:.3f}'),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel('min_samples', fontsize=10)
        ax.set_ylabel('min_cluster_size', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
 
    
        for i in range(len(min_cluster_sizes)):
            for j in range(len(min_samples_range)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=8, color='gray')
                else:
                    vmax = np.nanmax(matrix)
                    color = 'white' if val > vmax * 0.6 else 'black'
                    ax.text(j, i, fmt.format(val), ha='center', va='center',
                            fontsize=8, color=color, fontweight='bold')
 
    fig.suptitle(f'HDBSCAN hyperparameter tuning — {level_name}\n',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname_fig = f'{prefix}hdbscan_tuning_{level_name}.png'
    plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Graph saved : {fname_fig}")
 
    print(f"\n── Summary tuning HDBSCAN {level_name} ──")
    print(f"  n_clusters : {int(mat_clusters.min())} → {int(mat_clusters.max())}")
    print(f"  noise      : {mat_noise.min():.1f}% → {mat_noise.max():.1f}%")
 
    best_idx = np.unravel_index(np.nanargmax(mat_sil), mat_sil.shape)
    print(f"\n  Best Silhouette : {mat_sil[best_idx]:.4f}")
    print(f"    → min_cluster_size={min_cluster_sizes[best_idx[0]]}, "
          f"min_samples={min_samples_range[best_idx[1]]}")
    print(f"    → n_clusters={int(mat_clusters[best_idx])}, "
          f"noise={mat_noise[best_idx]:.1f}%")
 
    
    valid_mask = mat_clusters >= 2
    if valid_mask.any():
        noise_valid = np.where(valid_mask, mat_noise, np.inf)
        min_noise_idx = np.unravel_index(np.argmin(noise_valid), mat_noise.shape)
        print(f"\n  Less noise (with ≥2 clusters) : {mat_noise[min_noise_idx]:.1f}%")
        print(f"    → min_cluster_size={min_cluster_sizes[min_noise_idx[0]]}, "
              f"min_samples={min_samples_range[min_noise_idx[1]]}")
        print(f"    → n_clusters={int(mat_clusters[min_noise_idx])}")
 
    return df_tuning


def get_hdbscan_labels(X, min_cluster_size=5, min_samples=1):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return hdb.fit_predict(X)
 
 
def make_clinical_overlay(y, name, ax, X_2d, mask=None, noise_mask=None):
   
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
 

    no_data = ~mask
    if no_data.any():
        ax.scatter(X_2d[no_data, 0], X_2d[no_data, 1],
                   c='lightgrey', s=8, alpha=0.3, linewidths=0, label='N/A')
 
   
    le = LabelEncoder()
    y_valid = y[mask]
    y_enc   = le.fit_transform(y_valid.astype(str))
    classes = le.classes_
 
    cmap = plt.cm.get_cmap('tab10', len(classes))
    sc = ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=y_enc, cmap=cmap, vmin=0, vmax=len(classes)-1,
                    s=10, alpha=0.8, linewidths=0)
 
  
    handles = [mpatches.Patch(color=cmap(i), label=str(cls))
               for i, cls in enumerate(classes)]
    ax.legend(handles=handles, fontsize=6, loc='upper right',
              markerscale=0.8, framealpha=0.7)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.2)
 
 
def make_cluster_plot(labels, ax, X_2d, title):
    
    unique_labels = sorted(set(labels))
    n_clusters    = len([l for l in unique_labels if l != -1])
    cmap          = plt.cm.get_cmap('tab20', max(n_clusters, 1))
 

    noise = labels == -1
    if noise.any():
        ax.scatter(X_2d[noise, 0], X_2d[noise, 1],
                   c='lightgrey', s=8, alpha=0.3, linewidths=0, label='Noise')
 

    for idx, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(idx)], s=10, alpha=0.8, linewidths=0)
 
    n_noise = int(noise.sum())
    ax.set_title(f'{title}\n({n_clusters} clusters, {n_noise} noise)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.2)
 
# =============================================================================
# MAIN FUNCTION
# =============================================================================
 
def plot_2d_visualizations(levels, results, y_pam50, y_survival, y_stage, dfs, cfg,
                            level_name='level3',
                            hdbscan_min_cluster_size=5, hdbscan_min_samples=1,
                            tsne_perplexity=30, umap_n_neighbors=15, umap_min_dist=0.1,
                            random_state=42, prefix=''):
   
    X = levels[level_name]
    n_patients = X.shape[0]
 
  
    hdb_labels = get_hdbscan_labels(X, hdbscan_min_cluster_size, hdbscan_min_samples)
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    print(f"\n── HDBSCAN {level_name} : {n_clusters} clusters, "
          f"{int(sum(hdb_labels==-1))} noise ({sum(hdb_labels==-1)/n_patients*100:.1f}%) ──")
 

    first_df   = next(iter(dfs.values()))
    stage_col  = cfg['stage_col']
    y_stage_953 = first_df[stage_col].values  
    stage_mask  = ~pd.isnull(y_stage_953)
    y_stage_full = y_stage_953.astype(object)
    y_stage_full[~stage_mask] = np.nan
 
   
    clinical = {
        'PAM50'   : (y_pam50,    np.ones(n_patients, dtype=bool)),
        'Survival': (y_survival, np.ones(n_patients, dtype=bool)),
        'Stage'   : (y_stage_full, stage_mask),
    }
 
  
    print("  Computing PCA...", end=' ', flush=True)
    X_pca = PCA(n_components=2, random_state=random_state).fit_transform(X)
    print("✓")
 
    print("  Computing tSNE...", end=' ', flush=True)
    X_tsne = TSNE(n_components=2, perplexity=tsne_perplexity,
                  random_state=random_state, n_jobs=-1).fit_transform(X)
    print("✓")
 
    print("  Computing UMAP...", end=' ', flush=True)
    X_umap = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors,
                       min_dist=umap_min_dist,
                       random_state=random_state).fit_transform(X)
    print("✓")
 
    reductions = {
        'PCA' : X_pca,
        'tSNE': X_tsne,
        'UMAP': X_umap,
    }
 

    for method_name, X_2d in reductions.items():
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
 
  
        make_cluster_plot(hdb_labels, axes[0], X_2d,
                          title=f'HDBSCAN clusters')
 
   
        for ax, (var_name, (y_var, mask)) in zip(axes[1:], clinical.items()):
            make_clinical_overlay(y_var, var_name, ax, X_2d, mask=mask)
 
        fig.suptitle(f'{method_name} — {level_name} embeddings  ',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = f'{prefix}viz2d_{method_name.lower()}_{level_name}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   Saved: {fname}")
 


# =============================================================================
# STEP 8 — Statistical Tests (paired t-test)
# =============================================================================

def run_statistical_tests(results, levels, y_pam50, k_fixed, prefix=''):
    print("\n" + "="*60)
    print("STATISTICAL TESTS (paired t-test)")
    print("="*60)

    rows = []

    # ── Silhouette ──
    level_comparisons = [
        ('level2', 'level0'),
        ('level2', 'level1'),
        ('level3', 'level2'),
        ('level3', 'level0'),
    ]

    print("\n── Silhouette : Level comparisons (KMeans) ──")
    for lv_a, lv_b in level_comparisons:
        sil_a = results['kmeans'][lv_a][k_fixed].get('sil_runs', [])
        sil_b = results['kmeans'][lv_b][k_fixed].get('sil_runs', [])
        if len(sil_a) > 1 and len(sil_b) > 1:
            t_stat, p_val = stats.ttest_rel(sil_a, sil_b)
            sig = ' significant' if p_val < 0.05 else ' not significant'
            print(f"  {lv_a} vs {lv_b}: t={t_stat:.4f}, p={p_val:.4f} → {sig}")
            rows.append({
                'Comparison'  : f'{lv_a} vs {lv_b}',
                'Metric'      : 'Silhouette',
                'Test'        : 'paired t-test',
                'mean_A'      : round(np.mean(sil_a), 4),
                'mean_B'      : round(np.mean(sil_b), 4),
                't_statistic' : round(t_stat, 4),
                'p_value'     : round(p_val, 4),
                'significant' : p_val < 0.05
            })
        else:
            print(f"  {lv_a} vs {lv_b}: sil_runs not available")

    # ── Silhouette Level 3 ──
    algo_comparisons = [
        ('hdbscan', 'kmeans'),
        ('hdbscan', 'spectral'),
        ('gmm',     'kmeans'),
    ]

    print("\n── Silhouette : Algo comparisons on Level 3 ──")
    for algo_a, algo_b in algo_comparisons:

        def get_sil_mean(algo, level, k):
            if algo == 'hdbscan':
                return results[algo][level].get('sil', np.nan)
            else:
                return results[algo][level][k].get('sil', np.nan)

        if algo_a == 'hdbscan' or algo_b == 'hdbscan':
            sil_a = get_sil_mean(algo_a, 'level3', k_fixed)
            sil_b = get_sil_mean(algo_b, 'level3', k_fixed)
            print(f"  {algo_a} vs {algo_b}: HDBSCAN is deterministic — no paired t-test possible")
            rows.append({
                'Comparison'  : f'{algo_a} vs {algo_b} (level3)',
                'Metric'      : 'Silhouette',
                'Test'        : 'N/A — HDBSCAN deterministic',
                'mean_A'      : round(sil_a, 4) if not np.isnan(sil_a) else None,
                'mean_B'      : round(sil_b, 4) if not np.isnan(sil_b) else None,
                't_statistic' : None,
                'p_value'     : None,
                'significant' : None
            })
            continue

        sil_a = results[algo_a]['level3'][k_fixed].get('sil_runs', [])
        sil_b = results[algo_b]['level3'][k_fixed].get('sil_runs', [])
        if len(sil_a) > 1 and len(sil_b) > 1:
            t_stat, p_val = stats.ttest_rel(sil_a, sil_b)
            sig = 'significant' if p_val < 0.05 else 'not significant'
            print(f"  {algo_a} vs {algo_b} (level3): t={t_stat:.4f}, p={p_val:.4f} → {sig}")
            rows.append({
                'Comparison'  : f'{algo_a} vs {algo_b} (level3)',
                'Metric'      : 'Silhouette',
                'Test'        : 'paired t-test',
                'mean_A'      : round(np.mean(sil_a), 4),
                'mean_B'      : round(np.mean(sil_b), 4),
                't_statistic' : round(t_stat, 4),
                'p_value'     : round(p_val, 4),
                'significant' : p_val < 0.05
            })
        else:
            print(f"  {algo_a} vs {algo_b}: sil_runs not available")

    # ── ARI : Level 2 vs Level 3 (KMeans) ──
    print("\n── ARI : Level 2 vs Level 3 (KMeans) ──")
    all_labels_l2 = results['kmeans']['level2'][k_fixed].get('all_labels', [])
    all_labels_l3 = results['kmeans']['level3'][k_fixed].get('all_labels', [])
    if len(all_labels_l2) > 1 and len(all_labels_l3) > 1:
        aris_l2 = [adjusted_rand_score(y_pam50, lbl) for lbl in all_labels_l2]
        aris_l3 = [adjusted_rand_score(y_pam50, lbl) for lbl in all_labels_l3]
        t_stat, p_val = stats.ttest_rel(aris_l2, aris_l3)
        sig = 'significant' if p_val < 0.05 else 'not significant'
        print(f"  level2 vs level3 ARI: t={t_stat:.4f}, p={p_val:.4f} → {sig}")
        rows.append({
            'Comparison'  : 'level2 vs level3 (ARI)',
            'Metric'      : 'ARI',
            'Test'        : 'paired t-test',
            'mean_A'      : round(np.mean(aris_l2), 4),
            'mean_B'      : round(np.mean(aris_l3), 4),
            't_statistic' : round(t_stat, 4),
            'p_value'     : round(p_val, 4),
            'significant' : p_val < 0.05
        })
    else:
        print("  all_labels not available for level2 or level3")

    # Export CSV
    df_tests = pd.DataFrame(rows)
    fname = f'{prefix}statistical_tests.csv'
    df_tests.to_csv(fname, index=False)
    print(f"\n Statistical tests saved in '{fname}'")

    return df_tests


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
 
    cfg   = CONFIG
    seeds = RANDOM_SEEDS
 
    # 1. Load
    dfs = load_data(cfg)
 
    # 2. Extract
    X_levels, X_stage, y_pam50, y_survival, y_stage, y_stage_binary, clinical_vars = extract_arrays(dfs, cfg)
 
    # 3. Normalization
    levels, levels_stage = normalize(X_levels, X_stage)
 
    # 4. Elbow
    plot_elbow(levels,       cfg['k_range'], 'Elbow – PAM50/Survival', f"{cfg['output_prefix']}elbow_pam50.png")
    plot_elbow(levels_stage, cfg['k_range'], 'Elbow – Stage',          f"{cfg['output_prefix']}elbow_stage.png")
 
    # 5. Clustering
    print("\n── Clustering PAM50/Survival ──")
    results       = run_clustering(levels,       cfg['k_range'], cfg, seeds)
    print("\n── Clustering Stage ──")
    results_stage = run_clustering(levels_stage, cfg['k_range'], cfg, seeds)
 
    # 6. Metrics
    print("\n── Metrics PAM50/Survival ──")
    print_metrics(results,       levels,       cfg['k_range'])
    print("\n── Metrics Stage ──")
    print_metrics(results_stage, levels_stage, cfg['k_range'])
 
    # 7. ARI / NMI
    print_ari_nmi(results, levels, y_pam50, cfg['k_fixed'])
    save_ari_nmi_to_csv(results, levels, y_pam50, cfg['k_fixed'], 
                    f"{cfg['output_prefix']}ari_nmi_results.csv")
 
    # ARI/NMI vs Stage binary
    print_ari_nmi_stage(results, levels, y_stage_binary, 4)
    save_ari_nmi_stage_to_csv(results, levels, y_stage_binary, 4,
                               f"{cfg['output_prefix']}ari_nmi_stage_binary_results.csv")

    
    # Heatmap ARI/NMI vs Stage binary
    mask_stage = ~np.isnan(y_stage_binary)
    y_stage_bin_clean = y_stage_binary[mask_stage].astype(int)

    algo_names  = list(results.keys())
    level_names = list(levels.keys())
    ari_matrix = np.zeros((len(algo_names), len(level_names)))
    nmi_matrix = np.zeros((len(algo_names), len(level_names)))

    for i, algo in enumerate(algo_names):
        for j, name in enumerate(level_names):
            entry = results[algo][name][4] if algo != 'hdbscan' else results[algo][name]
            all_labels = entry.get('all_labels', [entry['labels']])
            aris = [adjusted_rand_score(y_stage_bin_clean, lbl[mask_stage]) 
                    for lbl in all_labels]
            nmis = [normalized_mutual_info_score(y_stage_bin_clean, lbl[mask_stage]) 
                    for lbl in all_labels]
            ari_matrix[i, j] = np.mean(aris)
            nmi_matrix[i, j] = np.mean(nmis)

    cmap = LinearSegmentedColormap.from_list('clin', ['#f7fbff', '#2171b5'])
    ylabels = [ALGO_LABELS.get(a, a) for a in algo_names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, matrix, title in [
        (axes[0], ari_matrix, 'ARI vs Stage Binary (Early/Late)'),
        (axes[1], nmi_matrix, 'NMI vs Stage Binary (Early/Late)'),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0)
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels(level_names, fontsize=9)
        ax.set_yticks(range(len(algo_names)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)
        for i in range(len(algo_names)):
            for j in range(len(level_names)):
                val = matrix[i, j]
                color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
    fig.suptitle('Clinical coherence — ARI & NMI vs Stage Binary (k=4)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"{cfg['output_prefix']}stage_binary_heatmap_ari_nmi.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {fname}")




    # Statistical tests
    df_tests = run_statistical_tests(results, levels, y_pam50, k_fixed=4,
                                      prefix=cfg['output_prefix'])
    

    # 8. Chi-square
    #print_chi2(results, results_stage, levels, levels_stage, clinical_vars, y_stage, cfg['k_fixed'])
    print_chi2(results, results_stage, levels, levels_stage, clinical_vars, y_stage, 4)
 
    # 9. Visualizations
    plot_bic_aic(results, levels, cfg['k_range'], cfg['output_prefix'])
    #plot_pca_clusters(results, levels, level_name='level3', k_fixed=cfg['k_fixed'], prefix=cfg['output_prefix'])
    plot_pca_clusters(results, levels, level_name='level3',
                      k_fixed=4, prefix=cfg['output_prefix'], y_pam50=y_pam50)
    plot_kmeans_levels_comparison(results, levels, k_fixed=cfg['k_fixed'], prefix=cfg['output_prefix'])
    #plot_silhouette_db_barplot(results, levels, k_fixed=cfg['k_fixed'], prefix=cfg['output_prefix'])
    plot_silhouette_db_barplot(results, levels, k_fixed=4, prefix=cfg['output_prefix'])
 
    #plot_ari_nmi_heatmap(results, levels, y_pam50, k_fixed=cfg['k_fixed'], prefix=cfg['output_prefix'])
    plot_ari_nmi_heatmap(results, levels, y_pam50, k_fixed=4, prefix=cfg['output_prefix'])
 
    #plot_confusion_clusters_pam50(results, levels, y_pam50, k_fixed=cfg['k_fixed'],prefix=cfg['output_prefix'])
    plot_confusion_clusters_pam50(results, levels, y_pam50, k_fixed=4,
                                  prefix=cfg['output_prefix'],level_name='level3')

   
    save_results_to_csv(results, results_stage, levels, levels_stage, cfg['k_range'], f"{cfg['output_prefix']}clustering_results.csv")

    print("\n\n" + "="*60)
    print("VISUALIZATION 2D — PCA / tSNE / UMAP")
    print("="*60)
 
    plot_2d_visualizations(
        levels        = levels,
        results       = results,
        y_pam50       = y_pam50,
        y_survival    = y_survival,
        y_stage       = y_stage,
        dfs           = dfs,
        cfg           = cfg, 
        level_name    = 'level3',
        hdbscan_min_cluster_size = 5,
        hdbscan_min_samples      = 1,
        tsne_perplexity          = 30,
        umap_n_neighbors         = 15,
        umap_min_dist            = 0.1,
        random_state             = 42,
        prefix        = cfg['output_prefix']
    )


 
   
 