import pandas as pd
import statsmodels.api as sm
import polars as pl
import numpy as np
import re

def group_predicted(predicted, group, score):
    "Average MAP scores by grouping variable"
    df = predicted \
        .group_by(group) \
        .agg(pl.col(score).mean().alias(score)) \
        .sort(score) \
        .to_pandas() 
        
    return df

def adjust_map_scores(df, X, model, score="Class_1"):
    """ Adjust MAP scores for count"""
    df["Score"] = df[score] - model.predict(X) \
        + model.params["const"] \
        + model.params["NCells"] * 1000
        
    for mut in df["Mutations"].unique():
       df["Score"] = df["Score"] + (df["Mutations"] == mut) * model.params[mut]
        
    return df


def fit_size_model(df, score="Class_1"):
    """Fit MAP score cell count adjustment model"""
    X = pd.get_dummies(
        df[['NCells', 'Mutations']], 
        prefix="",
        prefix_sep="",
        dtype=float
    )

    X = sm.add_constant(X)
    model = sm.OLS(df[score], X).fit()
    return model, X

def conformal_prediction_sets_core(cal_df, eval_df, quantile=0.9):
    """Generates multiclass conformal prediction sets.
    Assumes columns prob_{i} for each class i and column Label for true class.
    """
    # Find all class probability columns
    prob_cols = [col for col in cal_df.columns if re.match(r'prob_.*', col)]
    classes = [col.replace('prob_', '') for col in prob_cols]
    
    # Compute nonconformity scores for calibration set
    def get_nonconformity(row):
        return 1 - row[f'prob_{row["Label"]}']
    
    cal_df = cal_df.copy()
    cal_df['nonconformity'] = cal_df.apply(get_nonconformity, axis=1)
    
    quantile = np.clip(quantile * (len(cal_df) + 1) / len(cal_df), 0, 1) 
    q = cal_df['nonconformity'].quantile(quantile)
    
    # For each eval row, compute nonconformity for all classes
    def eval_pred_set(row):
        return [cl for cl in classes if 1 - row[f'prob_{cl}'] <= q]
        
    eval_df = eval_df.copy()
    eval_df['PredictionSet'] = eval_df.apply(eval_pred_set, axis=1)
    
    # Covered column: true label in prediction set
    eval_df['Covered'] = eval_df.apply(
        lambda row: str(row['Label']) in row['PredictionSet'], axis=1
    )
   
    return eval_df[['prob_1', 'CellLines', 'Label', 'PredictionSet', 'Covered']]


def conformal_prediction_sets(df, quantile=0.9, random_state=42):
    """Wrapper to run cross conformal prediction."""
    # Split datasets
    np.random.seed(random_state)
    celllines = df['CellLines'].unique()
    n = len(celllines)
    idx = np.random.permutation(n)
    half = n // 2
    
    cal_idx, eval_idx = idx[:half], idx[half:]
    cal_lines, eval_lines = celllines[cal_idx], celllines[eval_idx]
    
    # Compute conformal prediction scores for each dataset
    cal_df = df[df['CellLines'].isin(cal_lines)]
    eval_df = df[df['CellLines'].isin(eval_lines)]
    out1 = conformal_prediction_sets_core(cal_df, eval_df, quantile)
    out2 = conformal_prediction_sets_core(eval_df, cal_df, quantile)
    
    out_df = pd.concat([out1, out2], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=['CellLines']).reset_index(drop=True)
    out_df = out_df.sort_values('prob_1', ascending=False).reset_index(drop=True)
    return out_df