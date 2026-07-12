"""
Task 3 –  Patient Retrieval
=======================================================
Validate embedding quality through a clinically intuitive retrieval paradigm: given a query patient, are their nearest neighbors in embedding space clinically
similar?
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from lifelines.utils import concordance_index 

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
    'time' : 'time',
    'event' : 'event',
    'extra_clinical_cols' : ['T_stage', 'N_stage', 'M_stage']
}
RANDOM_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
k = [5,10,20]


# =============================================================================
# Step 0 — Loading and Preparing the Data
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
    time_col = config['time']
    event_col = config['event']
 
    y_pam50    = first_df[pam50_col].values
    y_survival = first_df[surv_col].values
    y_time = first_df[time_col].values  
    y_event = first_df[event_col].values 
 
    print("\nMissing values (level0):")
    print(first_df[[pam50_col, surv_col, stage_col]].isnull().sum())
 
  
    X_levels = {}
    for level, df in dfs.items():

        prefix   = config['embedding_prefix'].format(level[-1])
        emb_cols = [c for c in df.columns if c.startswith(prefix)]

        X_levels[level] = df[emb_cols].values
 

    stage_mask = ~first_df[stage_col].isna()
    y_stage    = first_df.loc[stage_mask, stage_col].values

    # Stage binary : Early (0,1,2) vs Late (3,4)
    y_stage_binary_full = np.full(len(first_df), np.nan)
    y_stage_binary_full[stage_mask] = np.where(y_stage <= 2, 0, 1)
    y_stage_binary = np.where(y_stage <= 2, 0, 1)
 
    X_stage = {}
    for level, df in dfs.items():
        prefix   = config['embedding_prefix'].format(level[-1])
        emb_cols = [c for c in df.columns if c.startswith(prefix)]
        X_stage[f'{level}_stage'] = df.loc[stage_mask, emb_cols].values
 
    
    extra = {col: first_df[col].values
             for col in config['extra_clinical_cols']
             if col in first_df.columns}
    clinical_vars = {
        'PAM50'   : y_pam50,
        'Survival': y_survival,
        **extra
    }
 
    return X_levels, X_stage, y_pam50, y_survival, y_time, y_event, y_stage, y_stage_binary, clinical_vars

# =============================================================================
# — Normalization
# =============================================================================

def normalize(X_levels, X_stage):
    
    levels       = {name: StandardScaler().fit_transform(X) for name, X in X_levels.items()}
    levels_stage = {name: StandardScaler().fit_transform(X) for name, X in X_stage.items()}
    return levels, levels_stage


# =============================================================================
# Step 1 — Cosine Similarity
# =============================================================================

def pairwise_cosine(levels, k_values):
    results_neighbors = {}
    similarity_matrix ={}


    for name, X in levels.items():
        print(f"Processing {name}...")
        sim_matrix = cosine_similarity(X)
        nb_patients = X.shape[0]

        similarity_matrix[name] = sim_matrix
        results_neighbors[name]={}

        for patient in range(nb_patients):
            sims = sim_matrix[patient].copy()
            sims[patient] =  -1

            results_neighbors[name][patient]={}
            for k in k_values :
                top_k = np.argsort(sims)[::-1][:k]
                results_neighbors[name][patient][k]=top_k.tolist()

    return results_neighbors, similarity_matrix

# =============================================================================
# Step 2 — Calculation of Precision@k for PAM50
# =============================================================================

def precision_k(levels,y_PAM50,k_values,neighbors):
    results ={}

    if not isinstance(y_PAM50, np.ndarray):
        y_PAM50 = np.array(y_PAM50)

    for name, X in levels.items():
        nb_patients = X.shape[0]
        results[name]={}

        for k in k_values :
            precisions=[]

            for patient in range(nb_patients):

                query_label = y_PAM50[patient]
                neighbours = neighbors[name][patient][k]
                correct = np.sum(y_PAM50[neighbours] == query_label)
                precision = correct / k
                precisions.append(precision)

            results[name][k] = np.mean(precisions)

    return results

def plot_precision_k(results, y_pam50, prefix=''):

    baseline = pd.Series(y_pam50).value_counts(normalize=True).max()

    plt.figure(figsize=(8, 5))

    for level, scores in results.items():

        k_values = sorted(scores.keys())
        precisions = [scores[k] for k in k_values]

        plt.plot(k_values, precisions, marker="o", label=level)

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f'Random retrieval ({baseline:.2f})'
    )

    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision@K across K values and model levels (Higher is better)")

    plt.legend()

    fname = f'{prefix}curves_precision@K.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {fname}")

    plt.show()

# =============================================================================
# Step 3 — Precision of Survival
# =============================================================================

def survival_precision(neighbors, y_event, k_values):
    
    results = {}

    if not isinstance(y_event, np.ndarray):
        y_event = np.array(y_event)

    for level_name in neighbors.keys():
        print(f"Processing {level_name}...")
        n_patients = len(neighbors[level_name])
        results[level_name] = {}

        for k in k_values:
            precisions = []

            for patient in range(n_patients):
                neighbour_indices = neighbors[level_name][patient][k]
                query_label = y_event[patient]
                correct = np.sum(y_event[neighbour_indices] == query_label)
                precision = correct / k
                precisions.append(precision)

            results[level_name][k] = np.mean(precisions)

    return results

def plot_survival_precision(results, y_survival, prefix=''):

    baseline = 0.5

    plt.figure(figsize=(8, 5))

    for level, scores in results.items():

        k_values = sorted(scores.keys())
        index = [scores[k] for k in k_values]

        plt.plot(k_values, index, marker="o", label=level)

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f'Random baseline ({baseline:.2f})'
    )

    plt.xlabel("K")
    plt.ylabel("Precision Survival")
    plt.title("Precision Survival across K values and model levels (Higher is better)")

    plt.legend()

    fname = f'{prefix}curves_precision_survival.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f" Saved: {fname}")

    plt.show()

# =============================================================================
# Step 4 — Stage Concordance
# =============================================================================

def stage_concordance(levels,neighbors,y_stage,k_values):
    results ={}

    if not isinstance(y_stage, np.ndarray):
        y_stage = np.array(y_stage)

    for name, X in levels.items():
        nb_patients = X.shape[0]
        results[name]={}

        for k in k_values :
            scores = []

            for patient in range(nb_patients):

                neighbours = neighbors[name][patient][k]
                diff = np.abs(y_stage[patient] -y_stage[neighbours])

                scores.append(np.mean(diff))

            results[name][k] = np.mean(scores)

    return results

def plot_stage_concordance(results, y_stage, prefix=''):

    y_stage = np.array(y_stage)

    baseline = np.mean(
        np.abs(y_stage[:, None] - y_stage[None, :])
    )

    plt.figure(figsize=(8, 5))

    for level, scores in results.items():

        k_values = sorted(scores.keys())
        values = [scores[k] for k in k_values]

        plt.plot(k_values, values, marker="o", label=level)

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f'Random baseline ({baseline:.2f})'
    )

    plt.xlabel("K")
    plt.ylabel("Mean absolute stage concordance")
    plt.title("Stage Concordance across K values and model levels (lower is better)")

    plt.legend()

    fname = f'{prefix}curves_stage_concordance.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {fname}")

    plt.show()


def stage_precision_binary(levels, neighbors, y_stage_binary, k_values):
    results = {}

    if not isinstance(y_stage_binary, np.ndarray):
        y_stage_binary = np.array(y_stage_binary)

    for name, X in levels.items():
        nb_patients = X.shape[0]
        results[name] = {}

        for k in k_values:
            precisions = []

            for patient in range(nb_patients):
                query_label = y_stage_binary[patient]
                neighbours = neighbors[name][patient][k]

                correct = np.sum(y_stage_binary[neighbours] == query_label)
                precision = correct / k
                precisions.append(precision)

            results[name][k] = np.mean(precisions)

    return results

def plot_stage_precision_binary(results, y_stage_binary, prefix=''):
    baseline = pd.Series(y_stage_binary).value_counts(normalize=True).max()

    plt.figure(figsize=(8, 5))

    for level, scores in results.items():
        k_values = sorted(scores.keys())
        precisions = [scores[k] for k in k_values]
        plt.plot(k_values, precisions, marker="o", label=level)

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f'Random retrieval ({baseline:.2f})'
    )

    plt.xlabel("K")
    plt.ylabel("Stage Precision@K (Early/Late)")
    plt.title("Stage Binary Precision@K across K values and model levels (Higher is better)")
    plt.legend()

    fname = f'{prefix}curves_stage_precision_binary.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    cfg   = CONFIG
    k_values = k
 
    # 1. Load
    dfs = load_data(cfg)
 
    # 2. Extract
    X_levels, X_stage, y_pam50, y_survival, y_time, y_event, y_stage, y_stage_binary, clinical_vars = extract_arrays(dfs, cfg)
 
    # 3. Normalization
    levels, levels_stage = normalize(X_levels, X_stage)

    # 4. Cosine Similarity
    neighbors, similarity_matrix = pairwise_cosine(levels,k_values)

    first_level = list(neighbors.keys())[0]
    first_patient = 0
    print(f"\n Example for {first_level}, patient {first_patient}:")
    for k in k_values:
        neighbours = neighbors[first_level][first_patient][k]
        print(f"  k={k} → neighbors : {neighbours[:5]}...")  

    # 5. Precision@K

    precision_results = precision_k(levels, y_pam50, k_values, neighbors)

    for level in precision_results:
        print(f"\n{level}:")
        for k, prec in precision_results[level].items():
            print(f"  Precision@{k}: {prec:.4f}")

    plot_precision_k(precision_results,y_pam50)

    # 6. Precision Survival

    survival_precision_results = survival_precision(neighbors,y_event,k_values)
    for level in survival_precision_results:
        print(f"\n{level}:")
        for k, ind in survival_precision_results[level].items():
            print(f"  Precision for Survival {k}: {ind:.4f}")
    
    plot_survival_precision(survival_precision_results,y_survival)

    # 7. Stage Concordance
    neighbors_stage, similarity_matrix_stage = pairwise_cosine(levels_stage,k_values)
    stage_results = stage_concordance(levels_stage, neighbors_stage, y_stage, k_values)

    for level in stage_results:
        print(f"\n{level}:")
        for k, stg in stage_results[level].items():
            print(f"  Stage Concordance for {k}: {stg:.4f}")

    plot_stage_concordance(stage_results,y_stage)

    # 8. Stage Precision Binary (Early vs Late)
    stage_binary_results = stage_precision_binary(levels_stage, neighbors_stage, 
                                                     y_stage_binary, k_values)

    for level in stage_binary_results:
        print(f"\n{level}:")
        for k, prec in stage_binary_results[level].items():
            print(f"  Stage Binary Precision@{k}: {prec:.4f}")

    plot_stage_precision_binary(stage_binary_results, y_stage_binary)