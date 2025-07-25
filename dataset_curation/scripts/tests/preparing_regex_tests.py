from pandas import DataFrame

def extract_examples(df: DataFrame, n_rows: int, seed: int = 1):
    out = dict()
    for col in df.columns:
        sampled = sample(df, n_rows, col, seed)
        sampled = zip(sampled, ["<EXPECTED_OUTPUT>"]*len(sampled))
        out[col] = dict(sampled)
    return out

def sample(df: DataFrame, n_rows: int, of_col: "str", seed: int = 1) -> list:
    mask = df[of_col].notna()
    total = df[mask][of_col]
    subset = total.sample(n=n_rows, random_state=seed)
    return subset


if __name__ == "__main__":
    pass
