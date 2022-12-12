
def lagged_correlation(df, target, key, t):
    """
    Compute lag correlation between target and key df column
    """

    lagged_correlation = df[target].corr(df[key].shift(+t))
    print("type(lagged_correlation):", type(lagged_correlation))
    print("lagged_correlation(t=%g):" % t, lagged_correlation)


