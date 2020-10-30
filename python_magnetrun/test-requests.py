#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import getpass
import sys
import os
import re
import datetime

if sys.stdin.isatty():
    password = getpass.getpass('Using getpass: ')
else:
    print( 'Using readline' )
    password = sys.stdin.readline().rstrip()

print( 'Read: ', password )

import requests
import requests.exceptions

def getMagnetRecord(magnetID):
    """get records for a given magnetID"""
    
    global Magnets
    global missingIDs
    global url_files
    global magnet_record_save
    
    print("MagnetID=%s" % magnetID)
    records = []
    if not magnetID in Magnets.keys():
        Magnets[magnetID] = records

    # To get files for magnetID
    params_files = (
        ('ref', magnetID),
    )

    params_links = (
        ('ref', magnetID),
        ('link', ''),
    )

    # Get all files for a given MagnetID: 'M15061601 aka ref dict entry
    r = s.get(url=url_files, params=params_files)
    print( "data:", r.encoding, r.url)
    if r.status_code != 200:
        print("error %d loading %s" % (r.status_code, url_files) )
        sys.exit(1)
    # for f in r.text.split('<br>'):
    #    if f and not '~' in f :
    #        print( "\t", f )

    # Get link to files for a given MagnetID: 'M15061601 aka ref dict entry
    # Bug: Broken data output
    # this url returns all the records for the site attached to M15061601 when running the script
    r = s.get(url=url_files, params=params_links)
    print( "data:", r.encoding, r.url )
    if r.status_code != 200:
        print("error %d loading %s" % (p.status_code, url_files) )
        sys.exit(1)

    for f in r.text.split('<br>'):
        if f and not '~' in f :
            replace_str='<a href='+'\''+url_downloads+'?file='

            data = f.replace(replace_str,'').replace('</a>','') .replace('\'>',': ').split(': ')
            link = data[0].replace(' ','%20')
            site = link.replace('../../../','')
            site = re.sub('/.*txt','',site)

            tformat="%Y.%m.%d - %H:%M:%S"
            # print( "timestamp: " , data[1].replace('.txt','') )
            timestamp = datetime.datetime.strptime(data[1].replace('.txt',''), tformat) # shall be a real timestamp

            # Download a specific file
            params_downloads = 'file=%s&download=1' % link
            # print("downloads:", params_downloads)
            d = s.get(url=url_downloads, params=params_downloads)
            if d.status_code != 200:
                print("error %d download %s" % (d.status_code, url_downloads) )
                sys.exit(1)
            # print( "data:", d.url )

            lines = d.text.split('\n')[0] # get 1st line
            lines_items = lines.split('\t')
            #print("lines=", lines, "len(items)=%d" % len(lines_items), "item=", lines_items)
            actual_id = None
            if len(lines_items) == 2:
                actual_id = lines_items[1] #re.sub('\n.*','',actual_id)

            if not actual_id:
                print("%s: no name defined for Magnet" % link)
                # watch out: Overwrites the file if the file exists.
                # If the file does not exist, creates a new file for writing.
                # shall change the name of the file to: [site_]magnetid_timestamp.txt
                filename = link.replace('../../../','')
                filename = filename.replace('/','_').replace('%20','-')
                fo = open(filename, "w", newline='\n')
                fo.write(d.text)
                fo.close()
            else:
                if actual_id != magnetID:
                    missingIDs.append(actual_id)
                    if not actual_id in Magnets.keys():
                        print("Create a new entry: ", actual_id)
                        Magnets[actual_id] = []
                    Magnets[actual_id].append( (timestamp, site, link) )
                    print( "\t**", "timestamp=%s site=%s Mid=%s (item=%d) **" % (timestamp, site, actual_id, len(Magnets[actual_id])) )
                    
                else:
                    Magnets[magnetID].append( (timestamp, site, link) )

            if magnet_record_save:
                # watch out: Overwrites the file if the file exists.
                # If the file does not exist, creates a new file for writing.
                # shall change the name of the file to: [site_]magnetid_timestamp.txt
                filename = link.replace('../../../','')
                filename = filename.replace('/','_').replace('%20','-')
                fo = open(filename, "w", newline='\n')
                fo.write(d.text)
                fo.close()

# shall check if host ip up and running
base_url="http://147.173.83.216/site/sba/pages"
url_logging=base_url + "/" + "login.php"
url_downloads=base_url + "/" + "courbe.php"
url_files=base_url + "/" + "getfref.php"
url_status=base_url + "/" + "Etat.php"

# Fill in your details here to be posted to the login form.
payload = {
    'email': 'christophe.trophime@lncmi.cnrs.fr',
    'password': password
}

# Magnets
magnet_record_save=False
Magnets = dict()
Status = dict()
missingIDs = []

# Use 'with' to ensure the session context is closed after use.
with requests.Session() as s:
    p = s.post(url=url_logging, data=payload)
    # print the html returned or something more intelligent to see if it's a successful login page.
    print( "connect:", p.url, p.status_code )
    # check return status: if not ok stop
    if p.status_code != 200:
        print("error %d logging to %s" % (p.status_code, url_logging) )
        sys.exit(1)

    # Perform some webscrapping to get all declared magnetids
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    import lxml.html as lh
    page = s.get(url=url_status)
    if page.status_code != 200 or page.url != url_status:
        print("cannot logging to %s" % url_logging)
        sys.exit(1)
    print( "etat:", page.url )
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)
    # from Etat.php source
    # table : id datatable, tr id=row, thead, tbody, <td class="sorting_1"...>
    # Parse data that are stored between <tbody>..</tbody> of HTML
    tr_elements = doc.xpath('//tbody')
    # print("detected tables:", tr_elements)

    #Create empty list
    Mids=[]
    i = 0

    #For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        #print( '%d:"%s"'%(i,name) )
        td_elements = doc.xpath('//td')
        j = 0
        # get date ID status comment from sub element
        for d in t:
            j+=1
            name = d.text_content()
            if j == 2:
                # print( '\t%d:%s' % (j, name) )
                Mids.append(name)
            if j == 3:
                Status[Mids[-1]] = name

    Mids = sorted(set(Mids)) #uniq only: list(set(Mids))
    print( "MagnetIDs found: ", Mids) # uniq

    for magnetID in Mids:
        getMagnetRecord(magnetID)

    # sort records list by 4th element:
    #unsorted_list.sort(key=lambda x: x[3])
    #sorted(unsorted_list, key = lambda x: int(x[3]))

    # check records for each missingID
    while ( len(missingIDs) != 0 ):
        check_missingIDs = set(missingIDs)
        missingIDs.clear()
        print("check missingIDs")
        for magnetID in check_missingIDs:
            if not magnetID in Status.keys():
                Status[magnetID] = "missingref"
            getMagnetRecord(magnetID)
    
print("\nMagnets: ")
for magnet in Magnets.keys():
    Magnets[magnet] = list(set(Magnets[magnet]))
    Magnets[magnet].sort(key=lambda x: x[0])
    records = Magnets[magnet]
    print("** %s: status=%s, records=%d" % ( magnet, Status[magnet], len(records) ) )

for magnet in Magnets.keys():
    print("** %s: status=%s" % ( magnet, Status[magnet] ) )
    for record in Magnets[magnet]:
        print( record )
