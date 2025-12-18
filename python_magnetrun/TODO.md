# claw1test

- init flow velocity from IH and IB
- update velocity at each timestep
- add gauges to store $T(x_i, t)$
- finalize test
- what to do to tell Cooling H and Cooling B??

# heatexchanger
- make findh a method

# HMagnets
- download MAGConfiles?
- dowload a given record

# magnetdata
- add method to remove pikes
- add method to smooth data
- method to remove identical columns or at least Icoili with $i=2,..,14$

# test-request

# DB: Tables

- Magnets
- GObjects
- MagnetID_[rapid]records
- MagnetID_GObjects

# Rework mrecord object (offline operational data)

- get record files from pigbrother and pupitre
- get bprofiles from profiles

- link with users DB

- mrecord: pupitre data + pigbrother data + eventually profiles
- add signatures for mrecord

- with analysis-refactor: create dict (per day) with
  - overview data: archive + pupitres + defaults + ...,


# Remove ICoil in dataframe

```python
keys = df.columns.values.tolist()
max_tap=0
for i in range(1,args.nhelices+1):
    ukey = "Ucoil%d" % i
    # print ("Ukey=%s" % ukey, (ukey in keys) )
    if ukey in keys:
        max_tap=i
if args.check:
    #print ("max_tap=%d" % max_tap)
    print (max_tap == args.nhelices)
    exit(0)

print ("max_tap=%d" % max_tap)
for i in range(2,max_tap):
    ikey = "Icoil%d" % i
    del df[ikey]

if "Icoil16" in keys:
    del df["Icoil16"]
```

- use probes for magnets objects in msite object (see magnetgeo) instead

# Analysis-refactor
- performance:
  - [] avoid loading data several times
  - [] best strategies for reducing memory (cache?) and speedup
  - [] measure memory footprint

- plotting:
  - [] downsampling for plot
  - [x] simple downsampling percent
  - [] more downsampling sophisticated method (see email)
    
- improvements
  - [x] get t_offset from tdms

- signature
  - [] save signature for reference per overview data
  - [] save timerange when several pupitre attached to overview
  - [] save signatures.times and signatures.values as a csv file with column names ['time', key]
  - [] save an approx of teb  as a csv file with column names ['time', key]
  
- water flow params
  - [x] compute best hysteresis model for debitbrut data with increasing levels (up to 4)
  - [] save besthysteris model
  - [] water_flows params per pupitre file
  - [] add water_flow and hysteresis model to signature per pupitre?

- lag and DTW
  - [] analyse DTW between pupitre and overview data
  - what conclusions?

- defauts/spikes
  - [x] display defauts and spikes on overview plots
  - [] search for the actual defauts/spikes details - see log from pigbrother (see logtdms)
  - [] see test-anomalies.py to display defauts/spikes - from details
  - [] try to perform automatic detection of defauts/spikes
