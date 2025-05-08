import pandas as pd
import statsmodels.api as sm
import polars as pl

def group_predicted(predicted, group, score):
    "Average MAP scores by grouping variable"
    df = predicted \
        .group_by(group) \
        .agg(pl.col(score).mean().alias(score)) \
        .sort(score) \
        .to_pandas() 
        
    return df

def adjust_map_scores(df, X, model):
    """ Adjust MAP scores for count"""
    df["Score"] = df["Ypred"] - model.predict(X) \
        + model.params["const"] \
        + model.params["NCells"] * 1000
        
    for mut in df["Mutations"].unique():
       df["Score"] = df["Score"] + (df["Mutations"] == mut) * model.params[mut]
        
    return df


def fit_size_model(df):
    """Fit MAP score cell count adjustment model"""
    X = pd.get_dummies(
        df[['NCells', 'Mutations']], 
        prefix="",
        prefix_sep="",
        dtype=float
    )

    X = sm.add_constant(X)
    model = sm.OLS(df['Ypred'], X).fit()
    return model, X
