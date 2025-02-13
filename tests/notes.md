= records

```bash
python -m python_magnetrun.analysis  pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_240509-*.tdms --key Référence_GR1 --show --synchronize
```

- `M10_2019.05.30---17:22:17.txt`: missing data for `Flow`
- `M10_2019.06.26---23:07:35.txt`: strange data for `tsb`
- `M10_2020.10.23---20.10.41.txt`: example plateau at 30 teslas

- `M9_Overview_240509-1634.tdms`: example anomalies on `Interne6` and `Interne7`
- `M9_Overview_240716-*.tdms`: example with default `tdms` files

Petit bilan de la dernière semaine de manip RMN sur M9 concernant l'instabilité de la polyhélice,

Jour : durée - nombre de détections*
mercredi 31/07 : 8:30 - 278
jeudi : 8:00 - 115
vendredi : 7:30 - 190
samedi : 7:00 - 351
dimanche 04/08 : 4:20 - 936
*détection = Pic en tension sur un des couples d'hélice (principalement 6-7 dans notre cas), cf. rapport illustratif.

Examples anomalies in `pigbrotherdata/Fichiers_Data/M9/Fichiers_Spike/M9_Spikes_*.tdms`

= lag correlation

Not working properly why???

= Gaussian peaks

see https://github.com/emilyripka/BlogRepo/blob/master/181119_PeakFitting.ipynb
= multiple plots

```python
import matplotlib.pyplot as plt

# to get multiple figure
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)
```

= zscore

```python
def zscore(x, window):
    r = x.rolling(window=window)
    m = r.median().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

### s.rolling(window=5).mean()
zscore(df[key], args.window).plot()
```



