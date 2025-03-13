= records

== problematic files

M9_Overview_200629-1003.tdms only one archive M9_Archive_200629-1503.tdms starting 5h later!!!

lag is not functionning for this case

== with anomalies

```bash
python -m python_magnetrun.analysis-refactor  pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_240509-*.tdms --key Référence_GR1 --show --synchronize
```

- `M10_2019.05.30---17:22:17.txt`: missing data for `Flow`
- `M10_2019.06.26---23:07:35.txt`: strange data for `tsb`
- `M10_2020.10.23---20.10.41.txt`: example plateau at 30 teslas

- `M9_Overview_240509-1634.tdms`: example anomalies on `Interne6` and `Interne7`
- `M9_Overview_240716-*.tdms`: example with default `tdms` files
- `M9_Overview_240511-1150.tdms`: trou de 10s dans M9_2024.05.11---11:50:20.txt ()

2024.05.11	11:50:40	...	
2024.05.11	11:50:50	...	

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

= plot enveloppe

```python
import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)  # Measured values
y_err = 0.1 * np.random.rand(100)  # Example uncertainties

# Calculate envelope
y_upper = y + y_err
y_lower = y - y_err

# Plot
plt.figure()
plt.plot(x, y, label="Measured Field", color="blue")  # Plot the measured values
plt.fill_between(x, y_lower, y_upper, color="lightblue", alpha=0.5, label="Uncertainty Envelope")  # Plot the envelope
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Measured Field with Uncertainty Envelope")
plt.legend()
plt.grid(True)
plt.show()
```

= timeseries index

* return the index of the element which equals 7 in myseries: Index(myseries).get_loc(7)
* get index from its positions?

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


= pupitre files

Rename after "manual" download:

```bash
for file in 2025.*.txt; do echo mv \"$file\" M9_$(echo "$file" | tr ' ' '-'| tr '_' ':'); done
```

