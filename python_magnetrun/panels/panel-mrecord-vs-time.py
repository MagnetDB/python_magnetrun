# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

import pandas as pd; import numpy as np; import matplotlib.pyplot as plt

from .python_magnetrun import MagnetRun

# data = pd.read_csv('./datatest.txt')
mrun = MagnetRun.fromtxt("M9", '../python_magnetsetup/data/mrecords/M9_2019.06.19---17:04:21.txt')
print("mrun:", mrun)
keys = mrun.getKeys()
print("keys:", keys)
print("type:", mrun.MagnetData.getType())

mrun.MagnetData.cleanupData()
print("keys:", mrun.getKeys())
mrun.MagnetData.addTime()
mrun.MagnetData.removeData('Date')
mrun.MagnetData.removeData('Time')
print("keys:", mrun.getKeys())

data = mrun.MagnetData.Data
# print("data:", type(data) )
# data.drop(['Date', 'Time'], axis=1)
# print("data:", data )

# shall make a timestamp colum from Date and Time
# data['date'] = data.date.astype('datetime64[ns]')
data = data.set_index('t')

data.tail()

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

# %matplotlib inline
def mpl_plot(avg, highlight):
    fig = Figure()
    FigureCanvas(fig) # not needed in mpl >= 3.1
    ax = fig.add_subplot()
    avg.plot(ax=ax)
    if len(highlight): highlight.plot(style='o', ax=ax)
    return fig

def find_outliers(variable='Field', window=30, sigma=10, view_fn=mpl_plot):
    avg = data[variable].rolling(window=window).mean()
    residual = data[variable] - avg
    std = residual.rolling(window=window).std()
    outliers = (np.abs(residual) > std * sigma)
    return view_fn(avg, avg[outliers])

import panel as pn
pn.extension()

import hvplot.pandas
import holoviews as hv

tap = hv.streams.PointerX(x=data.index.min())

def hvplot2(avg, highlight):
    line = avg.hvplot(height=300, width=500)
    outliers = highlight.hvplot.scatter(color='orange', padding=0.1)
    tap.source = line
    return (line * outliers).opts(legend_position='top_right')

@pn.depends(tap.param.x)
def table(x):
    # print("x=", x)
    index = np.abs((data.index - x).view(float)).argmin()
    # print("index:", index)
    return data.iloc[index]

import param
class MRecordData(param.Parameterized):
    variable  = param.Selector(objects=list(data.columns))
    window    = param.Integer(default=10, bounds=(1, 20))
    sigma     = param.Number(default=10, bounds=(0, 20))

    def view(self):
        return find_outliers(self.variable, self.window, self.sigma, hvplot2)

obj = MRecordData()

occupancy = pn.Column(
    pn.Row("## MRecord\nHover over the plot for more information.", obj.param),
    pn.Row(obj.view),
    pn.Row(table))
occupancy.show()
