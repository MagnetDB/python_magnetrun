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

import MRecord
import GObject
import HMagnet
import jsonpickle

def download(session, url_data, param, link=None, save=False, debug=False):
    """download """

    d = session.get(url=url_data, params=param)
    if debug:
        print("downloads:", d.url, d.status_code)
    if d.status_code != 200:
        print("error %d download %s" % (d.status_code, url_downloads) )
        sys.exit(1)
    d.raise_for_status()

    if save:
        filename = link.replace('../../../','')
        filename = filename.replace('/','_').replace('%20','-')
        print("save to %s" % filename)
        fo = open(filename, "w", newline='\n')
        fo.write(d.text)
        fo.close()

    return d.text

def getTable(session, url_data, index, indices, delimiter='//tbody', param=None, debug=False):
    """get table data from url_data"""

    # Perform some webscrapping to get all table data
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    if param is None:
        page = session.get(url=url_data)
    else:
        page = session.get(url=url_data, params=param)
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
    jid = None
    Mid = None
    Mjid = dict()
    Mdata = dict()

    if not tr_elements:
        if debug:
            print("page.text=", page.text, "**")
        return Mdata

    #For each row, store each first element (header) and an empty list
    for i,t in enumerate(tr_elements[0]):
        i+=1
        name=t.text_content()
        if debug:
            print( '%d:"%s"'%(i,name) )
        # get date ID status comment from sub element
        data = []
        for j,d in enumerate(t):
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
            Mjid[Mid] = jid

    # Mids = sorted(set(Mids)) #uniq only: list(set(Mids))
    if debug:
        print( "Data found: ", Mdata, "jid=", Mjid)
    return (Mdata, Mjid)

def getMagnetRecord(session, url_data, magnetID, Magnets, missingIDs, debug=False):
    """get records for a given magnetID"""

    if debug:
        print("MagnetID=%s" % magnetID)
    if not magnetID in Magnets.keys():
        Magnets[magnetID] = HMagnet.HMagnet(magnetID, 0, [], [], [], None, "Unknown", 0)

    # To get files for magnetID
    params_links = (
        ('ref', magnetID),
        ('link', ''),
    )

    r = session.get(url=url_data, params=params_links)
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
            html = download(session, url_downloads, param=params_downloads, link=link)

            lines = html.split('\n')[0] # get 1st line
            lines_items = lines.split('\t')

            actual_id = None
            if len(lines_items) == 2:
                actual_id = lines_items[1]

            if not actual_id:
                print("%s: no name defined for Magnet" % link)
            else:
                if actual_id != magnetID:
                    missingIDs.append(actual_id)
                    if not actual_id in Magnets.keys():
                        if debug:
                            print("Create a new entry: ", actual_id)
                        Magnets[actual_id] = HMagnet.HMagnet(actual_id, 0, [], [], [], None, "Unknown", 0)
                        MagnetRecords[actual_id] = []
                    Magnets[actual_id].addRecord( timestamp )
                    MagnetRecords[actual_id].append( MRecord.MRecord(timestamp, site, link) )
                    # print( "\t**", "timestamp=%s site=%s Mid=%s (item=%d) **" % (timestamp, site, actual_id, len(Magnets[actual_id])) )

                else:
                    Magnets[magnetID].addRecord( timestamp )
                    if not magnetID in MagnetRecords:
                        MagnetRecords[magnetID] = []
                    MagnetRecords[magnetID].append( MRecord.MRecord(timestamp, site, link) )
                    # print( "\t--", "timestamp=%s site=%s Mid=%s (item=%d) **" % (timestamp, site, actual_id, len(Magnets[magnetID])) )


if __name__ == "__main__":

    if sys.stdin.isatty():
        password = getpass.getpass('Using getpass: ')
    else:
        print( 'Using readline' )
        password = sys.stdin.readline().rstrip()

    print( 'Read: ', password )

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
    Magnets = dict()
    MagnetRecords = dict()
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

        # Get Magnets from Status page
        (Status, jid) = getTable(s, url_status, 2, [3])

        for i,magnetID in enumerate(Status): #Mids:
            getMagnetRecord(s, url_files, magnetID, Magnets, missingIDs)
            Magnets[magnetID].setStatus(Status[magnetID][-1])
            Magnets[magnetID].setIndex(jid[magnetID])
            
        # check records for each missingID
        while len(missingIDs) != 0 :
            check_missingIDs = set(missingIDs)
            missingIDs.clear()
            if debug:
                print("check missingIDs")
            for magnetID in check_missingIDs:
                if not magnetID in Status:
                    Status[magnetID] = "missingref"
                getMagnetRecord(s, url_files, magnetID, Magnets, missingIDs)
                Magnets[magnetID].setStatus(Status[magnetID][-1])

        if debug:
            print("\nMagnets: ")
        for magnet in Magnets:
            print("** %s: status=%s" % ( magnet, Magnets[magnet].getStatus() ) )
            if debug:
                print("loading helices for: ", magnet)

            params_helix = (
                ('ref', magnet),
            )

            hindices = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,19]
            res = getTable(s, url_helices, 1, hindices, param=params_helix)
            if res:
                helices = res[0]
                jid = res[1]
                if debug:
                    print("helices:", helices, "jid:", jid)

            for data in helices:
                # print("%s:" % data )
                for i in range(len(helices[data])-1):
                    materialID = re.sub('H.* / ','',helices[data][i])
                    if materialID != '-':
                        r = s.post(url_materials, data={ 'REF': materialID, 'compact:': 'on', 'formsubmit': 'OK' })
                        r.raise_for_status()
                        if debug:
                            print("post MaterialID: ", r.url, r.status_code)
                        html = lh.fromstring(r.text.encode(r.encoding))
                        conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
                        elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
                        if not materialID in Mats:
                            Mats[materialID] = GObject.GObject(materialID, 0,0,
                                                               {"sigma0":str(conductivity), "rpe": str(elasticlimit)},
                                                               "Helix", "Unknown")
                            Magnets[magnet].addGObject(materialID)

                        if debug:
                            print("Material: %s" % materialID,
                                  "Conductivity=", conductivity,
                                  "ElasticLimit=", elasticlimit)

                MAGconf = helices[data][-1]
                MAGconf.replace('  \t\t\t\t\t\t','')
                MAGconf.replace('\n',',')
                Magconf_files = MAGconf.split(' ')
                Magconf_files = [f for f in Magconf_files if f.endswith('.conf')]
                if debug:
                    print("MAGconfile=", Magconf_files, " **" )
                Magnets[magnet].setMAGfile(Magconf_files)


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
            print("Materials in main list:", len(refs)-1)

        for i,ref in enumerate(refs):
            # ref is lxml.etree._ElementUnicodeResult
            if not ref in Mats and not "finir" in ref:
                print("ref:", ref, type(ref), sigmas[i], elasticlimits[i])
                Mats[ref] = GObject.GObject(str(ref), 0,0,
                                            {"sigma0":str(sigmas[i]), "rpe":str(elasticlimits[i])},
                                            "Helix", "Unknown")

    print("\nSum up: ")
    print("\nMagnets:")
    for magnet in Magnets:
        # sort records list by timestamp:
        Magnets[magnet].getRecords().sort(key=lambda x: x.getTimestamp())
        print("** %s: status=%s, records=%d, helices=%d" % ( magnet,
                                                             Magnets[magnet].getStatus(),
                                                             len(Magnets[magnet].getRecords()),
                                                             len(Magnets[magnet].getGObjects()) ) )
        ## Broken to json:
        #try:
        if Magnets[magnet].getStatus() == "En service":
            fo = open(magnet + ".json", "w", newline='\n')
            fo.write(Magnets[magnet].to_json())
            fo.close()
            # except:
            #     print("fail to dump to json with deserialize")
            #     pass
            fo = open(magnet + "-pickle.json", "w", newline='\n')
            fo.write(jsonpickle.encode(Magnets[magnet]))
            fo.close()

    # for magnet in Magnets:
    #     print("** %s: status=%s" % ( magnet, Status[magnet] ) )
    #     for record in Magnets[magnet]:
    #         print( record )

    print("\nMaterials:")
    for mat in Mats:
        print(mat, ":", Mats[mat])
        fo = open(mat + ".json", "w", newline='\n')
        fo.write(jsonpickle.encode(Mats[mat]))
        fo.close()
