#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import getpass
import sys
# import os
import re
import datetime
import requests
import requests.exceptions
import lxml.html as lh

if sys.stdin.isatty():
    password = getpass.getpass('Using getpass: ')
else:
    print( 'Using readline' )
    password = sys.stdin.readline().rstrip()

print( 'Read: ', password )


# delimiter='//fieldset': when looking for MAGCONFILE.conf
def getTable(url_data, index=1, indices=[], delimiter='//tbody', param=None, debug=False):
    """get table data from url_data"""

    # Perform some webscrapping to get all table data
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    if param is None:
        page = s.get(url=url_data)
    else:
        page = s.get(url=url_data, params=param)
    if debug:
        print( "connect:", page.url, page.status_code )
    if page.status_code != 200 :
        print("cannot logging to %s" % url_data)
        sys.exit(1)
    page.raise_for_status()
        
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)

    # from php source
    # table : id datatable, tr id=row, thead, tbody, <td class="sorting_1"...>
    # Parse data that are stored between <tbody>..</tbody> of HTML
    tr_elements = doc.xpath(delimiter) # '//tbody')
    if debug:
        print("detected tables[delimiter=%s]:" % delimiter, tr_elements)

    #Create empty list ## Better to have a dict??
    Mid = None
    jid=None
    Mdata = dict()
    i = 0

    if not tr_elements:
        if debug:
            print("page.text=", page.text, "**")
        return Mdata

    #For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        if debug:
            print( '%d:"%s"'%(i,name) )
        j = 0
        # get date ID status comment from sub element
        data = []
        for d in t:
            j+=1
            jname = d.text_content()
            if debug:
                print( '\t%d:%s' % (j, jname) )
            if j == index:
                if param:
                    jid = jname[jname.find("(")+1:jname.find(")")]
                    # print("jid=", jid)
                Mid = re.sub(' (.*)','',jname)
            if j in indices:
                data.append(jname)
        # shall check wether key is already defined for sanity
        if Mid == "-" :
            print("%s index: no entry " % name)
        else:
            Mdata[Mid] = data
            # jid = i

    # Mids = sorted(set(Mids)) #uniq only: list(set(Mids))
    if debug:
        print( "Data found: ", Mdata, "jid=", jid)
    return (Mdata, jid)

def getMagnetRecord(url_data, magnetID, Magnets, missingIDs, magnet_record_save=False, debug=False):
    """get records for a given magnetID"""

    if debug:
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

    r = s.get(url=url_data, params=params_links)
    if debug:
        print( "data:", r.url, r.status_code, r.encoding )
    if r.status_code != 200:
        print("error %d loading %s" % (p.status_code, url_data) )
        sys.exit(1)
    r.raise_for_status()

    for f in r.text.split('<br>'):
        if f and not '~' in f :
            replace_str='<a href='+'\''+url_downloads+'?file='

            data = f.replace(replace_str,'').replace('</a>','') .replace('\'>',': ').split(': ')
            link = data[0].replace(' ','%20')
            site = link.replace('../../../','')
            site = re.sub('/.*txt','',site)

            tformat="%Y.%m.%d - %H:%M:%S"
            timestamp = datetime.datetime.strptime(data[1].replace('.txt',''), tformat)

            # Download a specific file
            params_downloads = 'file=%s&download=1' % link
            d = s.get(url=url_downloads, params=params_downloads)
            # print("downloads:", d.url, d.status_code)
            if d.status_code != 200:
                print("error %d download %s" % (d.status_code, url_downloads) )
                sys.exit(1)
            d.raise_for_status()
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
                        if debug: print("Create a new entry: ", actual_id)
                        Magnets[actual_id] = []
                    Magnets[actual_id].append( (timestamp, site, link) )
                    # print( "\t**", "timestamp=%s site=%s Mid=%s (item=%d) **" % (timestamp, site, actual_id, len(Magnets[actual_id])) )

                else:
                    Magnets[magnetID].append( (timestamp, site, link) )
                    # print( "\t--", "timestamp=%s site=%s Mid=%s (item=%d) **" % (timestamp, site, actual_id, len(Magnets[magnetID])) )

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
url_status=base_url + "/" + "Etat.php"
url_files=base_url + "/" + "getfref.php"
url_helices=base_url + "/" + "Aimant2.php"
url_materials=base_url + "/" + "Mat.php"


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
Mats = dict()
debug = False

# Use 'with' to ensure the session context is closed after use.
with requests.Session() as s:
    p = s.post(url=url_logging, data=payload)
    # print the html returned or something more intelligent to see if it's a successful login page.
    if debug:
        print( "connect:", p.url, p.status_code )
    # check return status: if not ok stop
    if p.status_code != 200:
        print("error %d logging to %s" % (p.status_code, url_logging) )
        sys.exit(1)
    p.raise_for_status()

    # test connection
    r = s.get(url=url_status)
    if r.url == url_logging:
        print("check connection failed: Wrong credentials" )
        sys.exit(1)
    
    # Perform some webscrapping to get all declared magnetids
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    (Status, jid) = getTable(url_status, 2, [3])
    # print("Status=", Status, "type=", type(Status))

    for magnetID in Status: #Mids:
        getMagnetRecord(url_files, magnetID, Magnets, missingIDs, magnet_record_save)
        # get jid from Aimant2.php ref=magnetID
        
    # check records for each missingID
    while len(missingIDs) != 0 :
        check_missingIDs = set(missingIDs)
        missingIDs.clear()
        if debug: print("check missingIDs")
        for magnetID in check_missingIDs:
            if not magnetID in Status:
                Status[magnetID] = "missingref"
            getMagnetRecord(url_files, magnetID, Magnets, missingIDs, magnet_record_save)

    if debug: print("\nMagnets: ")
    for magnet in Magnets:
        print("** %s: status=%s" % ( magnet, Status[magnet] ) )
        print("loading helices for: ", magnet)
            
        params_helix = (
            ('ref', magnet),
        )

        (helices, jid) = getTable(url_helices, 1, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,19], param=params_helix)
        for data in helices:
            # print("%s:" % data )
            for i in range(len(helices[data])-1):
                materialID = re.sub('H.* / ','',helices[data][i])
                if materialID != '-':
                    # Ref ou REF???
                    r = s.post(url_materials, data={ 'REF': materialID, 'compact:': 'on', 'formsubmit': 'OK' })
                    r.raise_for_status()
                    # print("post MaterialID: ", r.url, r.status_code)
                    html = lh.fromstring(r.text.encode(r.encoding))
                    conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
                    elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
                    if not materialID in Mats:
                        Mats[materialID] = [conductivity, elasticlimit]

                    #if debug:
                    print("Material: %s" % materialID,
                          "Conductivity=", conductivity,
                          "ElasticLimit=", elasticlimit)

            MAGconf = helices[data][-1]
            MAGconf.replace('  \t\t\t\t\t\t','')
            MAGconf.replace('\n',',')
            Magconf_files = MAGconf.split(' ')
            Magconf_files = [f for f in Magconf_files if f.endswith('.conf')]
            print("MAGconfile=", Magconf_files, " **" )

            
    print("\nMaterials Found:", len(Mats))
    print("\nMaterials")
    # Ref ou REF???
    r = s.post(url_materials, data={ 'compact:': 'on', 'formsubmit': 'OK' })
    r.raise_for_status()
    # print("post Material: ", r.url, r.status_code)
    html = lh.fromstring(r.text.encode(r.encoding))
    refs = html.xpath('//input[@name="REF"]/@value')
    sigmas = html.xpath('//input[@name="CONDUCTIVITE"]/@value')
    elasticlimits = html.xpath('//input[@name="LE"]/@value')
    if len(Mats.keys()) != len(refs)-1:
        print("Materials in main list:", len(refs)-1, len(sigmas)-1, len(elasticlimits)-1)

    for i,ref in enumerate(refs):
        if not ref in Mats and not "finir" in ref:
            # print("ref:", ref, sigmas[i], elasticlimits[i])
            Mats[ref] = [sigmas[i], elasticlimits[i]]
    # fo = open("materials.html", "w", newline='\n')
    # fo.write(r.text)
    # fo.close()

print("\nSum up: ")
print("\nMagnets:")
for magnet in Magnets:
    Magnets[magnet] = list(set(Magnets[magnet]))
    # sort records list by 4th element:
    Magnets[magnet].sort(key=lambda x: x[0])
    print("** %s: status=%s, records=%d" % ( magnet, Status[magnet], len(Magnets[magnet]) ) )

# for magnet in Magnets:
#     print("** %s: status=%s" % ( magnet, Status[magnet] ) )
#     for record in Magnets[magnet]:
#         print( record )

print("\nMaterials:")
for mat in Mats:
    print(mat, ":", Mats[mat])
