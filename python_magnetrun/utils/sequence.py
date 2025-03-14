"""Main module."""

def list_sequence(lst, seq):
    """Return sequences of seq in lst"""
    sequences = []
    count = 0
    len_seq = len(seq)
    upper_bound = len(lst)-len_seq+1
    for i in range(upper_bound):
        if lst[i:i+len_seq] == seq:
            count += 1
            sequences.append([i,i+len_seq])
    return sequences

# see:  https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
#from collections import defaultdict

def list_duplicates_of(seq,item):
    """Return sequences of duplicate adjacent item in seq"""
    start_at = -1
    locs = []
    sequences = []
    start_index = -1
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            end_index = locs[-1]
            sequences.append([start_index, end_index])
            # print("break end_index=%d" % end_index)
            break
        else:
            if not locs:
                # seq=[loc,0]
                start_index = loc
                # print( "item=%d, start: %d" % (item, loc) )
            else:
                if (loc-locs[-1]) != 1:
                    end_index = locs[-1]
                    sequences.append([start_index, end_index])
                    start_index = loc
                    # print( "item=%d, end: %d, new_start: %d" % (item, locs[-1], loc) )
            locs.append(loc)
            start_at = loc
    return sequences #locs


