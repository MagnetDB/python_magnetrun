# claw1test

init flow velocity from IH and IB
update velocity at each timestep
add gauges to store T(xi, t)
finalize test
what to do to tell Cooling H and Cooling B??

# heatexchanger
make findh a method

# HMagnets
download MAGConfiles?
dowload a given record

# magnetdata
add method to remove pikes
add method to smooth data
method to remove identical columns or at least Icoili with i=2,..,14

# test-request

# DB: Tables
Magnets
GObjects
MagnetID_[rapid]records
MagnetID_GObjects


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

# Set precise starting time for tdms

Because of downsampling applied to store pigbrother data:

* overview: 1/2 s
* archive: (1/120)/2 s
* default: 0

# Analysis-refactor

get Reference key from tdms
loop over reference key instead of passing reference key as argparse argument
save signature with flow_params and teb, debitbrut data