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

import panel as pn
pn.extension()

import hvplot.pandas
import holoviews as hv

from bokeh.sampledata.autompg import autompg_clean

# from holoviews import dim, opts
# from holoviews.plotting.bokeh.styles import (line_properties, fill_properties, text_properties)
# print("""
# Line properties: %s\n
# Fill properties: %s\n
# Text properties: %s
# """ % (line_properties, fill_properties, text_properties))

# http://holoviews.org/reference/elements/matplotlib/Scatter.html

import param
class MRecordData(param.Parameterized):
    xvariable  = param.Selector(objects=list(data.columns))
    yvariable  = param.Selector(objects=list(data.columns))

    def view(self):
        return hv.Scatter(data, self.xvariable, self.yvariable).opts(min_height=500, responsive=True)

obj = MRecordData()

occupancy = pn.Column(
    pn.Row("## MRecord\n", obj.param),
    pn.Row(obj.view))
occupancy.show()
