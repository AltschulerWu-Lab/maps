"""
Updated utility functions for i-MAP analysis.

This module provides utility functions that operate on dataframes with i-MAP scores
in a standard format:
- 'CellLines': observation cell line for each row
- 'True': true class integer value
- 'Class_0', 'Class_1', ..., 'Class_k': class predictions/probabilities

Functions include count corrections and conformal prediction set generation.
"""

from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import re
import polars as pl
from maps.archive.utils import adjust_map_scores, fit_size_model

# ============================================================================
# Count Correction Functions
# ============================================================================
def fit_count_correction_model(
    df: pd.DataFrame,
    cellline_to_mutation: Dict[str, str],
    cell_count_col: str = 'NCells',
    score_col: str = 'Class_1',
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Fit a linear model to correct i-MAP scores for cell count effects.
    
    The model includes cell count and mutation as predictors:
        score ~ NCells + Mutations
    
    Args:
        df: DataFrame with columns including CellLines, score_col, and cell_count_col
        cellline_to_mutation: Mapping from cell line names to mutation labels
        cell_count_col: Name of the column containing cell counts
        score_col: Name of the column containing the i-MAP score to correct
        
    Returns:
        Tuple of (fitted_model, design_matrix)
    """
    # Ensure we have mutations column
    df_model = df.copy()
    if 'Mutations' not in df_model.columns:
        df_model['Mutations'] = df_model['CellLines'].map(cellline_to_mutation)
    
    # Create design matrix with dummy variables for mutations
    X = pd.get_dummies(
        df_model[[cell_count_col, 'Mutations']],
        prefix="",
        prefix_sep="",
        dtype=float
    )
    
    X = sm.add_constant(X)
    
    # Fit OLS model
    y = df_model[score_col]
    model = sm.OLS(y, X).fit()
    
    return model, X


def apply_count_correction(
    df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    cellline_to_mutation: Dict[str, str],
    score_col: str = 'Class_1',
    cell_count_col: str = 'NCells',
    reference_count: float = 1000,
) -> pd.DataFrame:
    """
    Apply count correction to i-MAP scores using a fitted model.
    
    Adjusted score = original - predicted + intercept + NCells_effect * reference_count + mutation_effect
    
    Args:
        df: DataFrame with i-MAP scores
        model: Fitted statsmodels OLS model
        X: Design matrix used to fit the model
        cellline_to_mutation: Mapping from cell line names to mutation labels
        score_col: Name of the column containing the i-MAP score to correct
        cell_count_col: Name of the column containing cell counts
        reference_count: Reference cell count to adjust to (default: 1000)
        
    Returns:
        DataFrame with added 'Score_corrected' column
    """
    df_corrected = df.copy()
    
    # Ensure mutations column
    if 'Mutations' not in df_corrected.columns:
        df_corrected['Mutations'] = df_corrected['CellLines'].map(cellline_to_mutation)
    
    # Calculate adjusted scores
    predictions = model.predict(X)
    
    df_corrected['Score_corrected'] = (
        df_corrected[score_col] - predictions +
        model.params.get('const', 0) +
        model.params.get(cell_count_col, 0) * reference_count
    )
    
    # Add back mutation effects
    for mut in df_corrected['Mutations'].unique():
        if mut in model.params:
            mask = df_corrected['Mutations'] == mut
            df_corrected.loc[mask, 'Score_corrected'] += model.params[mut]
    
    return df_corrected


def correct_imap_scores(
    df: pd.DataFrame,
    cellline_to_mutation: Dict[str, str],
    cell_count_col: str = 'NCells',
    score_col: Union[str, list] = 'Class_1',
    reference_count: float = 1000,
) -> Tuple[pd.DataFrame, Union[sm.regression.linear_model.RegressionResultsWrapper, Dict]]:
    """
    Fit count correction model and apply it to i-MAP scores.
    
    For single score column: Returns corrected score in 'Score_corrected' column.
    For multiple score columns: Regresses each independently, applies softmax renormalization,
    and returns corrected probabilities in columns with '_corrected' suffix.
    
    Args:
        df: DataFrame with i-MAP scores and cell counts
        cellline_to_mutation: Mapping from cell line names to mutation labels
        cell_count_col: Name of the column containing cell counts
        score_col: Name(s) of column(s) containing i-MAP scores to correct.
                  Can be a single string or list of strings.
        reference_count: Reference cell count to adjust to (default: 1000)
        
    Returns:
        Tuple of (corrected_dataframe, fitted_model(s))
        - If score_col is str: returns (df, model)
        - If score_col is list: returns (df, dict_of_models)
        
    Examples:
        >>> # Single score correction
        >>> df_corr, model = correct_imap_scores(df, cellline_to_mutation, score_col='Class_1')
        
        >>> # Multiple scores with softmax renormalization
        >>> df_corr, models = correct_imap_scores(
        ...     df, cellline_to_mutation, 
        ...     score_col=['Class_0', 'Class_1', 'Class_2']
        ... )
    """
    # Handle single score column (backward compatibility)
    if isinstance(score_col, str):
        model, X = fit_count_correction_model(
            df, cellline_to_mutation, cell_count_col, score_col
        )
        
        df_corrected = apply_count_correction(
            df, model, X, cellline_to_mutation, score_col, cell_count_col, reference_count
        )
        
        return df_corrected, model
    
    # Handle multiple score columns with softmax renormalization
    score_cols = score_col  # Rename for clarity
    df_corrected = df.copy()
    
    # Ensure mutations column
    if 'Mutations' not in df_corrected.columns:
        df_corrected['Mutations'] = df_corrected['CellLines'].map(cellline_to_mutation)
    
    # Fit model for each score column
    models = {}
    corrected_logits = {}
    
    for col in score_cols:
        model, X = fit_count_correction_model(
            df, cellline_to_mutation, cell_count_col, col
        )
        models[col] = (model, X)
        
        # Calculate adjusted scores (logits before softmax)
        predictions = model.predict(X)
        
        corrected_logits[col] = (
            df_corrected[col] - predictions +
            model.params.get('const', 0) +
            model.params.get(cell_count_col, 0) * reference_count
        )
        
        # Add back mutation effects
        for mut in df_corrected['Mutations'].unique():
            if mut in model.params:
                mask = df_corrected['Mutations'] == mut
                corrected_logits[col].loc[mask] += model.params[mut]
    
    # Stack corrected logits and apply softmax
    logits_array = np.column_stack([corrected_logits[col].values for col in score_cols])
    
    # Softmax: exp(logits) / sum(exp(logits))
    exp_logits = np.exp(logits_array - logits_array.max(axis=1, keepdims=True))  # Numerical stability
    softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Add corrected probabilities to dataframe
    for i, col in enumerate(score_cols):
        df_corrected[f'{col}_corrected'] = softmax_probs[:, i]
    
    return df_corrected, models


# ============================================================================
# Conformal Prediction Functions
# ============================================================================

def _compute_nonconformity_scores(
    df: pd.DataFrame,
    true_col: str = 'True',
    class_prefix: str = 'Class_',
) -> pd.Series:
    """
    Compute nonconformity scores for conformal prediction.
    
    Nonconformity score = 1 - P(true_class)
    
    Args:
        df: DataFrame with true labels and class probabilities
        true_col: Name of the column containing true class labels
        class_prefix: Prefix for class probability columns
        
    Returns:
        Series of nonconformity scores
    """
    def get_nonconformity(row):
        true_class = int(row[true_col])
        prob_col = f'{class_prefix}{true_class}'
        return 1 - row[prob_col]
    
    return df.apply(get_nonconformity, axis=1)


def _get_conformal_threshold(
    nonconformity_scores: pd.Series,
    quantile: float = 0.9,
) -> float:
    """
    Compute conformal prediction threshold with finite-sample correction.
    
    Args:
        nonconformity_scores: Series of nonconformity scores from calibration set
        quantile: Desired coverage level (e.g., 0.9 for 90% coverage)
        
    Returns:
        Threshold value
    """
    n = len(nonconformity_scores)
    # Finite-sample correction
    adjusted_quantile = np.clip(quantile * (n + 1) / n, 0, 1)
    return nonconformity_scores.quantile(adjusted_quantile)


def _generate_prediction_set(
    row: pd.Series,
    threshold: float,
    class_prefix: str = 'Class_',
) -> list:
    """
    Generate prediction set for a single observation.
    
    Args:
        row: Series with class probabilities
        threshold: Nonconformity threshold
        class_prefix: Prefix for class probability columns
        
    Returns:
        List of class indices in the prediction set
    """
    class_cols = [col for col in row.index if col.startswith(class_prefix)]
    classes = [col.replace(class_prefix, '') for col in class_cols]
    
    # Include classes where nonconformity <= threshold
    # i.e., 1 - P(class) <= threshold, or P(class) >= 1 - threshold
    prediction_set = [
        cls for cls in classes
        if row[f'{class_prefix}{cls}'] >= (1 - threshold)
    ]
    
    return prediction_set


def conformal_prediction_sets_split(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantile: float = 0.9,
    true_col: str = 'True',
    class_prefix: str = 'Class_',
) -> pd.DataFrame:
    """
    Generate conformal prediction sets using calibration-evaluation split.
    
    Args:
        cal_df: Calibration DataFrame with columns CellLines, True, Class_0, Class_1, ...
        eval_df: Evaluation DataFrame with same structure
        quantile: Desired coverage level (e.g., 0.9 for 90% coverage)
        true_col: Name of the column containing true class labels
        class_prefix: Prefix for class probability columns
        
    Returns:
        DataFrame with columns CellLines, True, PredictionSet, Covered
    """
    # Compute nonconformity scores on calibration set
    cal_df = cal_df.copy()
    cal_df['nonconformity'] = _compute_nonconformity_scores(cal_df, true_col, class_prefix)
    
    # Get threshold
    threshold = _get_conformal_threshold(cal_df['nonconformity'], quantile)
    
    # Generate prediction sets for evaluation set
    eval_df = eval_df.copy()
    eval_df['PredictionSet'] = eval_df.apply(
        lambda row: _generate_prediction_set(row, threshold, class_prefix),
        axis=1
    )
    
    # Check coverage: true label in prediction set
    eval_df['Covered'] = eval_df.apply(
        lambda row: str(int(row[true_col])) in row['PredictionSet'],
        axis=1
    )
    
    return eval_df[['CellLines', true_col, 'PredictionSet', 'Covered']]


def conformal_prediction_sets_cross(
    df: pd.DataFrame,
    quantile: float = 0.9,
    true_col: str = 'True',
    class_prefix: str = 'Class_',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate conformal prediction sets using cross-conformal prediction.
    
    Splits cell lines into two halves, uses each half to calibrate for the other.
    
    Args:
        df: DataFrame with columns CellLines, True, Class_0, Class_1, ...
        quantile: Desired coverage level (e.g., 0.9 for 90% coverage)
        true_col: Name of the column containing true class labels
        class_prefix: Prefix for class probability columns
        random_state: Random seed for splitting
        
    Returns:
        DataFrame with columns CellLines, True, PredictionSet, Covered
    """
    # Split cell lines into two groups
    np.random.seed(random_state)
    celllines = df['CellLines'].unique()
    n = len(celllines)
    idx = np.random.permutation(n)
    half = n // 2
    
    cal_idx, eval_idx = idx[:half], idx[half:]
    cal_lines, eval_lines = celllines[cal_idx], celllines[eval_idx]
    
    # First split: calibrate on group 1, evaluate on group 2
    cal_df_1 = df[df['CellLines'].isin(cal_lines)]
    eval_df_1 = df[df['CellLines'].isin(eval_lines)]
    out_1 = conformal_prediction_sets_split(
        cal_df_1, eval_df_1, quantile, true_col, class_prefix
    )
    
    # Second split: calibrate on group 2, evaluate on group 1
    cal_df_2 = df[df['CellLines'].isin(eval_lines)]
    eval_df_2 = df[df['CellLines'].isin(cal_lines)]
    out_2 = conformal_prediction_sets_split(
        cal_df_2, eval_df_2, quantile, true_col, class_prefix
    )
    
    # Combine results
    out_df = pd.concat([out_1, out_2], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=['CellLines']).reset_index(drop=True)
    
    return out_df


def compute_conformal_threshold_grid(
    df: pd.DataFrame,
    quantiles: list = [0.75, 0.9],
    true_col: str = 'True',
    class_prefix: str = 'Class_',
) -> dict:
    """
    Compute conformal thresholds for multiple quantile levels.
    
    This is useful for visualizing uncertainty regions at different confidence levels.
    
    Args:
        df: Calibration DataFrame with columns True, Class_0, Class_1, ...
        quantiles: List of quantile levels to compute thresholds for
        true_col: Name of the column containing true class labels
        class_prefix: Prefix for class probability columns
        
    Returns:
        Dictionary mapping quantile to threshold value
    """
    # Compute nonconformity scores
    df = df.copy()
    df['nonconformity'] = _compute_nonconformity_scores(df, true_col, class_prefix)
    
    # Compute thresholds for each quantile
    thresholds = {}
    for q in quantiles:
        thresholds[q] = _get_conformal_threshold(df['nonconformity'], q)
    
    return thresholds


def generate_probability_simplex_grid(
    n_classes: int,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a grid of probability vectors on the (n_classes - 1)-simplex.
    
    This is useful for visualizing conformal prediction regions in probability space.
    
    Args:
        n_classes: Number of classes
        step: Step size for grid (default: 0.01)
        
    Returns:
        DataFrame where each row is a probability vector summing to 1
    """
    def generate_simplex_recursive(n_vars, total, step=0.01, current=[]):
        """Generate all combinations of n_vars non-negative values that sum to total."""
        if n_vars == 1:
            val = round(total, 2)
            if val >= -0.01:  # Allow small floating point errors
                yield current + [max(0, val)]
        else:
            # Generate values from 0 to total in steps
            for val in np.arange(0, total + step, step):
                val = round(val, 2)
                if val <= total + 0.01:
                    yield from generate_simplex_recursive(
                        n_vars - 1, total - val, step, current + [val]
                    )
    
    grid_data = []
    for probs in generate_simplex_recursive(n_classes, 1.0, step):
        row = {f'Class_{i}': probs[i] for i in range(n_classes)}
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)


def generate_conformal_regions(
    df: pd.DataFrame,
    quantiles: list = [0.75, 0.9],
    n_classes: Optional[int] = None,
    true_col: str = 'True',
    class_prefix: str = 'Class_',
    grid_step: float = 0.01,
) -> dict:
    """
    Generate conformal prediction regions in probability space.
    
    For each quantile and each class, generates the region of probability vectors
    where that class would be included in the prediction set (i.e., has high probability).
    
    The region for class i at quantile q includes all probability vectors where
    P(class_i) >= threshold, and the remaining probabilities sum to (1 - P(class_i)).
    This matches the conformal prediction criterion: a class is in the prediction set
    when its probability is high enough that 1 - P(class) <= threshold.
    
    Args:
        df: Calibration DataFrame
        quantiles: List of quantile levels
        n_classes: Number of classes (inferred from df if not provided)
        true_col: Name of the column containing true class labels
        class_prefix: Prefix for class probability columns
        grid_step: Step size for probability grid
        
    Returns:
        Dictionary with keys (quantile, class_idx) mapping to DataFrames of probability vectors
        
    Notes:
    -----
    This implementation matches the reference code in figures.ipynb which generates grids
    where a specific class has probability >= threshold (meaning it would be in the 
    prediction set), and the remaining classes can have any valid probability distribution.
    """
    # Infer number of classes if not provided
    if n_classes is None:
        class_cols = [col for col in df.columns if col.startswith(class_prefix)]
        n_classes = len(class_cols)
    
    # Compute thresholds
    thresholds = compute_conformal_threshold_grid(df, quantiles, true_col, class_prefix)
    
    # Helper function to generate all combinations that sum to target
    def generate_simplex_recursive(n_vars, total, step=0.01, current=[]):
        """Generate all combinations of n_vars non-negative values that sum to total."""
        if n_vars == 1:
            val = round(total, 2)
            if val >= -0.01:  # Allow small floating point errors
                yield current + [max(0, val)]
        else:
            for val in np.arange(0, total + step, step):
                val = round(val, 2)
                if val <= total + 0.01:
                    yield from generate_simplex_recursive(
                        n_vars - 1, total - val, step, current + [val]
                    )
    
    # For each (quantile, class) pair, generate prediction region
    regions = {}
    for quantile, threshold in thresholds.items():
        for threshold_col in range(n_classes):
            grid_data = []
            remaining_cols = [i for i in range(n_classes) if i != threshold_col]
            
            # The threshold column must have probability >= threshold for class to be in prediction set
            # This follows from: nonconformity = 1 - P(class), and we want nonconformity <= threshold
            # Therefore: P(class) >= 1 - threshold
            for threshold_val in np.arange(1 - threshold, 1.0 + grid_step, grid_step):
                threshold_val = round(threshold_val, 2)
                remaining = round(1.0 - threshold_val, 2)
                
                # Generate all combinations for the remaining classes
                for vals in generate_simplex_recursive(n_classes - 1, remaining, grid_step):
                    probs = [0] * n_classes
                    probs[threshold_col] = threshold_val
                    for col_idx, val in zip(remaining_cols, vals):
                        probs[col_idx] = val
                    
                    grid_data.append({f'{class_prefix}{i}': probs[i] for i in range(n_classes)})
            
            regions[(quantile, threshold_col)] = pd.DataFrame(grid_data)
    
    return regions


# ============================================================================
# Helper Functions for Data Aggregation
# ============================================================================

def aggregate_scores_by_cellline(
    df: pd.DataFrame,
    score_cols: Optional[list] = None,
    agg_func: str = 'mean',
) -> pd.DataFrame:
    """
    Aggregate i-MAP scores by cell line.
    
    Useful when you have multiple observations per cell line and want to
    compute summary statistics.
    
    Args:
        df: DataFrame with CellLines column and score columns
        score_cols: List of columns to aggregate (if None, aggregates all Class_* columns)
        agg_func: Aggregation function ('mean', 'median', 'std', etc.)
        
    Returns:
        DataFrame with one row per cell line
    """
    if score_cols is None:
        score_cols = [col for col in df.columns if col.startswith('Class_')]
    
    # Include True and other metadata columns if present
    group_cols = ['CellLines']
    metadata_cols = ['True', 'Mutations']
    for col in metadata_cols:
        if col in df.columns:
            group_cols.append(col)
    
    agg_dict = {col: agg_func for col in score_cols}
    
    # For True and Mutations, take the first value (should be consistent)
    for col in metadata_cols:
        if col in group_cols and col != 'CellLines':
            agg_dict[col] = 'first'
    
    result = df.groupby('CellLines').agg(agg_dict).reset_index()
    
    return result


def group_predicted(predicted, group, score="Class_1"):
    "Average MAP scores by grouping variable"
    df = predicted \
        .group_by(group) \
        .agg(pl.col(score).mean().alias(score)) \
        .sort(score) \
        .to_pandas() 
        
    return df