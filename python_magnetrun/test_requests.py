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
import requests
import requests.exceptions
import lxml.html as lh

# import jsonpickle
from . import MRecord
from . import GObject
from . import HMagnet

def createSession(url_logging, payload, cert):
    """create a request session"""

    # import ssl
    # ssl.match_hostname = lambda cert, hostname: hostname == cert['subjectAltName'][0][1]

    if debug:
        print( "connect:", url_logging )
    p = s.post(url=url_logging, data=payload, verify=cert)
    # print the html returned or something more intelligent to see if it's a successful login page.
    if debug:
        print( "connect:", p.url, p.status_code )
    # check return status: if not ok stop
    if p.status_code != 200:
        print("error %d logging to %s" % (p.status_code, url_logging) )
        sys.exit(1)
    p.raise_for_status()
    return p

def download(session, url_data, cert, param, link=None, save=False, debug=False):
    """download """

    d = session.get(url=url_data, params=param, verify=cert)
    if debug:
        print("downloads:", d.url, d.status_code)
    if d.status_code != 200:
        print("error %d download %s" % (d.status_code, url_data) )
        sys.exit(1)
    d.raise_for_status()

    if save:
        filename = link.replace('../../../','')
        filename = filename.replace('/','_').replace('%20','-')
        # print("save to %s" % filename)
        fo = open(filename, "w", newline='\n')
        fo.write(d.text)
        fo.close()

    return d.text

# for M1:
# table fileTreeDemo_1, ul, <li  class="file ext_txt">, <a href=.., rel="filename" /a> </li>

def getTable(session, url_data, index, indices, cert, delimiter='//tbody', param=None, debug=False):
    """get table data from url_data"""

    # Perform some webscrapping to get all table data
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    if param is None:
        page = session.get(url=url_data, verify=cert)
    else:
        page = session.get(url=url_data, params=param, verify=cert)
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

def getMagnetRecord(session, url_data, magnetID, Magnets, missingIDs, cert, debug=False):
    """get records for a given magnetID"""

    if debug:
        print("MagnetID=%s" % magnetID)
    if not magnetID in Magnets.keys():
        Magnets[magnetID] = HMagnet.HMagnet(magnetID, 0, None, "Unknown", 0)

    # To get files for magnetID
    params_links = (
        ('ref', magnetID),
        ('link', ''),
    )

    r = session.get(url=url_data, params=params_links, verify=cert)
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
            html = download(session, url_downloads, cert, param=params_downloads, link=link)

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
                        Magnets[actual_id] = HMagnet.HMagnet(actual_id, 0, None, "Unknown", 0)
                        MagnetRecords[actual_id] = []
                    # Magnets[actual_id].addRecord( timestamp )
                    record = MRecord.MRecord(timestamp, site, link)
                    if not record in MagnetRecords[actual_id]:
                        # print("actual_id: %s - %s, %s %s" %(actual_id, timestamp, site, link) )
                        MagnetRecords[actual_id].append( record )

                else:
                    # Magnets[magnetID].addRecord( timestamp )
                    if not magnetID in MagnetRecords:
                        MagnetRecords[magnetID] = []

                    record = MRecord.MRecord(timestamp, site, link)
                    if not record in MagnetRecords[magnetID]:
                        # print("magnetID: %s - %s, %s %s" %(magnetID, timestamp, site, link) )
                        MagnetRecords[magnetID].append( record )


if __name__ == "__main__":
    import argparse
    from . import python_magnetrun
    import matplotlib
    import matplotlib.pyplot as plt
    
    requests.packages.urllib3.disable_warnings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="specify user")
    parser.add_argument("--server", help="specify server", default="https://data") #.grenoble.lncmi.local") # "https://147.173.83.216/site/sba/pages")
    parser.add_argument("--cert", help="specify server cert", default="lncmi-data.pem")
    parser.add_argument("--save", help="save files", action='store_true')
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
    args = parser.parse_args()

    if sys.stdin.isatty():
        password = getpass.getpass('Using getpass: ')
    else:
        print( 'Using readline' )
        password = sys.stdin.readline().rstrip()

    if not os.path.isfile(args.cert):
        print("failed to load server certificat: %s" % args.cert)
        sys.exit(1)

    # print( 'Read: ', password )

    # shall check if host ip up and running
    base_url=args.server
    url_logging=base_url + "/site/sba/pages/" + "login.php"
    url_downloads=base_url + "/site/sba/pages/" + "courbe.php"
    url_status=base_url + "/site/sba/pages/" + "Etat.php"
    url_files=base_url + "/site/sba/pages/" + "getfref.php"
    url_helices=base_url + "/site/sba/pages/" + "Aimant2.php"
    url_materials=base_url + "/site/sba/pages/" + "Mat.php"
    url_query=base_url + "/site/sba/vendor/jqueryFileTree/connectors/jqueryFileTree.php"


    # Fill in your details here to be posted to the login form.
    payload = {
        'email': args.user,
        'password': password
    }

    # Magnets
    Magnets = dict()
    MagnetRecords = dict()
    MagnetComps = dict()
    Status = dict()
    missingIDs = []
    Mats = dict()
    debug = args.debug

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = createSession(url_logging, payload, args.cert)

        # test connection
        r = s.get(url=url_status, verify=args.cert)
        if r.url == url_logging:
            print("check connection failed: Wrong credentials" )
            sys.exit(1)

        # Get Magnets from Status page
        (Status, jid) = getTable(s, url_status, 2, [3], args.cert)

        for i,magnetID in enumerate(Status): #Mids:
            getMagnetRecord(s, url_files, magnetID, Magnets, missingIDs, args.cert)
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
                getMagnetRecord(s, url_files, magnetID, Magnets, missingIDs, args.cert)
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
            res = getTable(s, url_helices, 1, hindices, args.cert, param=params_helix)
            helices = ()
            if res:
                helices = res[0]
                jid = res[1]
                if Magnets[magnet].getIndex() is None:
                    Magnets[magnet].setIndex(jid[magnet])
                if debug:
                    print("helices:", helices, "jid:", jid, Magnets[magnet].getIndex() )

            for data in helices:
                # print("%s:" % data )
                for i in range(len(helices[data])-1):
                    materialID = re.sub('H.* / ','',helices[data][i])
                    if debug:
                        print("%s:" % materialID )
                    if materialID != '-':
                        r = s.post(url_materials, data={ 'REF': materialID, 'compact:': 'on', 'formsubmit': 'OK' }, verify=args.cert)
                        r.raise_for_status()
                        # if debug:
                        #     print("post MaterialID: ", r.url, r.status_code)
                        html = lh.fromstring(r.text.encode(r.encoding))
                        conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
                        elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
                        if not materialID in Mats:
                            Mats[materialID] = GObject.GObject(materialID, 0,0,
                                                               {"sigma0":str(conductivity), "rpe": str(elasticlimit)},
                                                               "Helix", "Unknown")
                        #Magnets[magnet].addGObject(materialID)
                        if not magnet in MagnetComps:
                            MagnetComps[magnet] = []
                        MagnetComps[magnet].append(materialID)
                        if debug:
                            print("MagnetComps[%s].append(%s)" % (magnet,materialID) )
                
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


        print("\nMaterials")
        print("\nMaterials Found:", len(Mats))
        # Ref ou REF???
        r = s.post(url_materials, data={ 'compact:': 'on', 'formsubmit': 'OK' }, verify=args.cert)
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
                if debug:
                    print("ref:", ref, type(ref), sigmas[i], elasticlimits[i])
                Mats[ref] = GObject.GObject(str(ref), 0,0,
                                            {"sigma0":str(sigmas[i]), "rpe":str(elasticlimits[i])},
                                            "Helix", "Unknown")

        #try:
        print("=============================")
        for i in [1, 5, 7, 8 , 9, 10]:
            print("Loading txt files for M%d site" % i)
            sitename = "/var/www/html/M%d/" % i
            sitename = sitename.replace('/','%2F')
            # print("sitename=", sitename)
            
            r = s.post(url=url_query, data={ 'dir': sitename  , }, verify=args.cert)
            # print("r.url=", r.url)
            r.raise_for_status()
            # print("r.text=", r.text)
            tree = lh.fromstring(r.text)
            # print("tree:", tree)
            for tr in tree.xpath('//a'):
                if tr.text_content().endswith(".txt"):
                    print('tr:', tr.text_content() )
                    try:
                        tformat="%Y.%m.%d - %H:%M:%S"
                        timestamp = datetime.datetime.strptime(tr.text_content().replace('.txt',''), tformat)
                    except:
                        tformat="%Y.%m.%d - %H_%M_%S"
                        timestamp = datetime.datetime.strptime(tr.text_content().replace('.txt',''), tformat)
                        print("changed tformat: %s" % tr.text_content())
                        pass

                    link = "../../../M%d/%s" % (i,tr.text_content().replace(' ','%20'))

                    # print("MRecord: ", timestamp, "M%d" % i, link)
                    record = MRecord.MRecord(timestamp, "M%d" % i, link)
                    data = record.getData(s, url_downloads, args.cert, save=args.save)
                    # print("data=", data)
                    mrun = python_magnetrun.MagnetRun.fromStringIO("M%d"%i, data)
                    insert = mrun.getInsert()
                    # print("M%d: insert=%s file=%s" % (i, insert, tr.text_content()) )
                    if not insert in Magnets:
                        Magnets[insert] = HMagnet.HMagnet(insert, 0, None, "Unknown", 0)
                    if not insert in MagnetRecords:
                        MagnetRecords[insert] = []
                    if not record in MagnetRecords[insert]:
                        #print("addRecord: %s, %s, %s" % (insert, "M%d"%i, link) )
                        MagnetRecords[insert].append( record )
        print("=============================")
        # except:
        #     print( "Failed to perform jqueryFileTree" )
        #     pass
                      

    print("\nSum up: ")
    print("\nMagnets:")
    for magnet in Magnets:
        if not magnet in MagnetRecords:
            MagnetRecords[magnet] = []
        if not magnet in MagnetComps:
            MagnetComps[magnet] = []

        print("** %s: status=%s, records=%d, helices=%d" % ( magnet,
                                                             Magnets[magnet].getStatus(),
                                                             len(MagnetRecords[magnet]),
                                                             len(MagnetComps[magnet]) ) )

    print("\nMagnets in Operation:")
    for magnet in Magnets:
        ## Broken to json:
        #try:
        if Magnets[magnet].getStatus() == "En service":
            print("%s: " % magnet)
            if args.save:
                fo = open(magnet + ".json", "w", newline='\n')
                fo.write(Magnets[magnet].to_json())
                fo.close()
            for record in MagnetRecords[magnet]:
                ax = plt.gca()

                data = record.getData(s, url_downloads, args.cert, save=args.save)
                try:
                    mrun = python_magnetrun.MagnetRun.fromStringIO(record.getSite(), data)
                    mrun.plateaus(thresold=2.e-3, duration=10, save=args.save)
                except:
                    print("record: trouble with data from %s" % record.getLink())
                    pass

    print("\nMaterials:")
    for mat in Mats:
        print(mat, ":", Mats[mat])
        if args.save:
            fo = open(mat + ".json", "w", newline='\n')
            fo.write(Mats[mat].to_json())
            fo.close()
