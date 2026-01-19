import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from sklearn.pipeline import make_union

index = pd.date_range(start="2020-01-01", end="2020-01-05", inclusive="left", freq="H")
data = pd.DataFrame(index=index, data=[10] * len(index), columns=["value"])
data["date"] = index.date

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return X["value"].groupby(X["date"]).sum()

# Test with default output
set_config(transform_output="default")
result_default = make_union(MyTransformer()).fit_transform(data)
print("Default output shape:", result_default.shape)

# Test with pandas output
set_config(transform_output="pandas")
try:
    result_pandas = make_union(MyTransformer()).fit_transform(data)
    print("Pandas output shape:", result_pandas.shape)
except Exception as e:
    print("Error with pandas output:", str(e))
