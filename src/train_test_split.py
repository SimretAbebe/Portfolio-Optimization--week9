import pandas as pd

def chronological_split(series: pd.Series, split_date: str):
    """
    Split a pandas Series (or DataFrame indexed by dates) into a training set that ends
    before `split_date` and a test set that starts on or after that date.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    train = series[series.index < split_date]
    test = series[series.index >= split_date]

    print(f"[INFO] Train period: {train.index.min()} → {train.index.max()}")
    print(f"[INFO] Train size:   {len(train)} days")
    print(f"[INFO] Test period:  {test.index.min()} → {test.index.max()}")
    print(f"[INFO] Test size:    {len(test)} days")

    return train, test