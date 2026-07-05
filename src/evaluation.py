import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(actual, predicted, model_name="Model"):
    """Compute MAE, RMSE and MAPE and return them as a dict.
    Also prints a friendly summary.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"{model_name} performance:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%\n")
    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
    }


def compare_models(results_list):
    """Create a DataFrame comparing multiple model result dicts, sorted by RMSE."""
    df = pd.DataFrame(results_list)
    return df.sort_values("RMSE").reset_index(drop=True)
