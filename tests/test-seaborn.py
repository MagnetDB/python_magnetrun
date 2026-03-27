import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402

# df = sm.datasets.co2.load(as_pandas=True).data
df = sm.datasets.co2.load().data
df["month"] = pd.to_datetime(df.index).month
df["year"] = pd.to_datetime(df.index).year
sns.lineplot(x="month", y="co2", hue="year", data=df.query("year>1995"))
plt.show()
