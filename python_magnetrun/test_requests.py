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
import json
import requests
import requests.exceptions
import lxml.html as lh

from . import MRecord
from . import GObject
from . import HMagnet

def createSession(url_logging, payload):
    """create a request session"""

    if debug:
        print( "connect:", url_logging )
    p = s.post(url=url_logging, data=payload, verify=True)
    # print the html returned or something more intelligent to see if it's a successful login page.
    if debug:
        print( "connect:", p.url, p.status_code )
    # check return status: if not ok stop
    if p.status_code != 200:
        print("error %d logging to %s" % (p.status_code, url_logging) )
        sys.exit(1)
    p.raise_for_status()
    return p

def download(session, url_data, param, link=None, save=False, debug=False):
    """download """

    d = session.get(url=url_data, params=param, verify=True)
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

def getTable(session, url_data, index, indices, delimiter='//tbody', param=None, debug=False):
    """get table data from url_data"""

    # Perform some webscrapping to get all table data
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    if param is None:
        page = session.get(url=url_data, verify=True)
    else:
        page = session.get(url=url_data, params=param, verify=True)
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
        return (Mdata, Mjid)

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

def getMagnetRecord(session, url_data, magnetID, Magnets, save=False, debug=False):
    """get records for a given magnetID"""
    print(f'getMagnetRecord({magnetID})')
    
    if not magnetID in Magnets.keys():
        Magnets[magnetID] = HMagnet.HMagnet(magnetID, 0, None, "Unknown", 0)

    # To get files for magnetID
    params_links = (
        ('ref', magnetID),
        ('link', ''),
    )

    r = session.get(url=url_data, params=params_links, verify=True)
    if debug:
        print( f"data: url={r.url}, status={r.status_code}, encoding={r.encoding}, text={r.text}")
    if r.status_code != 200:
        print("error %d loading %s" % (p.status_code, url_data) )
        sys.exit(1)
    r.raise_for_status()

    for f in r.text.split('<br>'):
        if f and not '~' in f :
            replace_str='<a href='+'\''+url_downloads+'?file='

            data = f.replace(replace_str,'').replace('</a>','') .replace('\'>',': ').split(': ')
            link = data[0].replace(' ','%20')
            link = re.sub('<a?(.*?)file=', '', link,  flags=re.DOTALL)
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
            if len(lines_items) >= 2:
                actual_id = lines_items[1]
            if debug:
                print(f'{magnetID}: actual_id={actual_id}, site={site} link={link}, param={params_downloads}')
            
            if not actual_id:
                if debug: print("%s: no name defined for Magnet" % link)
            else:
                record = MRecord.MRecord(timestamp, site, link)
                data = record.getData(s, url_downloads, save)

                if actual_id != magnetID:
                    print(f"record: incoherent data magnetID {magnetID} actual_id: {actual_id} - {timestamp}, {site} {link}" )
                    # TO change magnetID in txt once downloaded
                    data = data.replace(actual_id,magnetID)
                    # overwrite data
                    if save:
                        filename = link.replace('../../../','')
                        filename = filename.replace('/','_').replace('%20','-')
                        # print("save to %s" % filename)
                        fo = open(filename, "w", newline='\n')
                        fo.write(data)
                        fo.close()

                # Magnets[magnetID].addRecord( timestamp )
                if not magnetID in MagnetRecords:
                    MagnetRecords[magnetID] = []

                if not record in MagnetRecords[magnetID]:
                    if debug: print(f"{magnetID}: {timestamp} - {site}, {link}" )
                    MagnetRecords[magnetID].append( record )
        else:
            if debug:
                print(f'getMagnetRecords({magnetID}): f={f}')


if __name__ == "__main__":
    import argparse
    from . import python_magnetrun
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="specify user")
    parser.add_argument("--server", help="specify server", default="https://srv-data-install.lncmi.cnrs.fr/")
    parser.add_argument("--save", help="save files", action='store_true')
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
    args = parser.parse_args()

    if sys.stdin.isatty():
        password = getpass.getpass('Using getpass: ')
    else:
        print( 'Using readline' )
        password = sys.stdin.readline().rstrip()

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
    Mats = dict()
    debug = args.debug

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = createSession(url_logging, payload)
        # test connection
        # print(f'url_status={url_status}')
        r = s.get(url=url_status, verify=True)
        if r.url == url_logging:
            print("check connection failed: Wrong credentials" )
            sys.exit(1)
        
        # Get Magnets from Status page
        (Status, jid) = getTable(s, url_status, 2, [3])
        if args.debug:
            print(f'Status: {Status}, jid={jid}')

        for i,magnetID in enumerate(Status): #Mids:
            getMagnetRecord(s, url_files, magnetID, Magnets, args.save, args.debug)
            # print(f'getMagnetRecord({magnetID}): {MagnetRecords[magnetID]}')
            Magnets[magnetID].setStatus(Status[magnetID][-1])
            Magnets[magnetID].setIndex(jid[magnetID])

        Parts = {}
        if debug:
            print("\nMagnets: ")
        for magnet in Magnets:
            print(f"** {magnet}: status={Magnets[magnet].getStatus()}")
            if debug:
                print(f"loading helices for: {magnet}")

            params_helix = (
                ('ref', magnet),
            )

            hindices = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,19]
            res = getTable(s, url_helices, 1, hindices, param=params_helix)
            helices = ()
            if res:
                helices = res[0]
                jid = res[1]
                if Magnets[magnet].getIndex() is None:
                    Magnets[magnet].setIndex(jid[magnet])
                if debug:
                    print(f"helices: {helices}, jid: {jid} index: {Magnets[magnet].getIndex()}")

            if not magnet in Parts:
                Parts[magnet]=[]
            if debug:
                print(f"{magnet}: jid={jid}, index={Magnets[magnet].getIndex()}")
            for data in helices:
                # print("%s:" % data )
                for i in range(len(helices[data])-1):
                    materialID = re.sub('H.* / ','',helices[data][i])
                    if debug:
                        print("%s:" % materialID )
                    if materialID != '-':
                        Parts[magnet].append([i,materialID.replace('H','MA')])
                        r = s.post(url_materials, data={ 'REF': materialID, 'compact:': 'on', 'formsubmit': 'OK' }, verify=True)
                        r.raise_for_status()
                        # if debug:
                        #     print("post MaterialID: ", r.url, r.status_code)
                        html = lh.fromstring(r.text.encode(r.encoding))
                        conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
                        elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
                        nuance = html.xpath('//input[@name="NUANCE"]/@value')[-1]
                        # print(materialID, nuance)
                        if not materialID in Mats:
                            Mats[materialID] = GObject.GObject(materialID, 0,0,
                                                               {"sigma0":str(conductivity), "rpe": str(elasticlimit), "nuance": nuance},
                                                               "helix", "Unknown")
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

        print("\nMsites")
        # ???
        
        print("\nMagnets:", len(Magnets))
        PartName = {}
        for magnet in Magnets:
            # print(magnet, type(Magnets[magnet]))
            carac = {'name':magnet.replace('M','M'),'status':Magnets[magnet].status}
            magconf = Magnets[magnet].MAGfile
            # print("parts:", Parts[magnet])
            if magconf:
                magconffile = magconf[0]
                carac['config'] = magconffile
            if Parts[magnet]:
                carac['parts'] = []
                for part in Parts[magnet]:
                    pname = part[-1].replace('MA','H')
                    pid = part[0]
                    carac['parts'].append(pname)
                    if not pname in PartName:
                        PartName[pname] = f"HL-31_H{pid}"
            mname = magnet.replace('M','M')
            print(mname, carac)
            if args.save:
                fo = open(f'{mname}.json', "w", newline='\n')
                fo.write(json.dumps(carac, indent=4))
                fo.close()
        
        print("\nMaterials")
        print("\nMaterials Found:", len(Mats))
        # Ref ou REF???
        r = s.post(url_materials, data={ 'compact:': 'on', 'formsubmit': 'OK' }, verify=True)
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
                                            "helix", "Unknown")

        print("\nMParts")
        for mat in Mats:
            # print(mat, type(Mats[mat]))
            key = mat.replace('MA','H')
            carac = {'name': mat.replace('MA','H'),
                     'type': Mats[mat].category,
                     'material': mat
                    }
            if key in PartName:
                carac['geometry'] = PartName[key]
            print(key, carac)
            if args.save:
                fo = open(f'{key}.json', "w", newline='\n')
                fo.write(json.dumps(carac, indent=4))
                fo.close()

        print("\nMaterials")
        for mat in Mats:
            carac = {'name':Mats[mat].name,
                     'description': '',
                     't_ref': 293,
                     'volumic_mass': 9e+3,
                     'specific_heat': 0,
                     'alpha': 3.6e-3,
                     'electrical_conductivity': Mats[mat].material['sigma0'],
                     'thermal_conductivity': 380,
                     'magnet_permeability': 1,
                     'young': 117e+9,
                     'poisson': 0.33,
                     'expansion_coefficient': 18e-6,
                     'rpe': Mats[mat].material['rpe']
                     }
            if 'nuance' in Mats[mat].material:
                carac['nuance'] = Mats[mat].material['nuance']
            print(mat, carac)
            if args.save:
                fo = open(f'{mat}.json', "w", newline='\n')
                fo.write(json.dumps(carac, indent=4))
                fo.close()
                      
        sites = {}
        for magnet in Magnets:
            housing_records={}
            if magnet in MagnetRecords:
                housing_records={}
                for record in MagnetRecords[magnet]:
                    filename = record.link.replace('../../../','')
                    filename = filename.replace('/','_').replace('%20','-')
                    housing = filename.split('_')[0]
                    if housing in housing_records:
                        housing_records[housing].append(filename)
                    else:
                        housing_records[housing] = [filename]

                sites[ f'{housing}_{magnet}' ] = housing_records[housing]

        print("\nSites:")
        for site in sites:
            housing = site.split('_')[0]
            carac = {'name': site, 'status': 'in_study'}
            print(site, carac)
            if args.save:
                fo = open(f'{housing}_{site}.json', "w", newline='\n')
                fo.write(json.dumps(carac, indent=4))
                fo.close()

        print("\nRecords:")
        for site in sites:
            for i,file in enumerate(sites[site]):
                # TODO save record
                carac = {'file':file, 'site':site}
                print(site, carac)
                if args.save:
                    fo = open(f'{site}_record{i}.json', "w", newline='\n')
                    fo.write(json.dumps(carac, indent=4))
                    fo.close()

        # print("\nMaterials:")
        # for mat in Mats:
        #     print(mat, ":", Mats[mat])
        #     if args.save:
        #         fo = open(mat + ".json", "w", newline='\n')
        #         fo.write(Mats[mat].to_json())
        #         fo.close()

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

    print("\nMagnets - Basic stats:")
    for magnet in Magnets:
        for record in MagnetRecords[magnet]:
            print(f"magnet={magnet}, record={record.getSite()}, link={record.getLink()},url_downloads={url_downloads}")
            data = record.getData(s, url_downloads, save=args.save)
            try:
                mrun = python_magnetrun.MagnetRun.fromStringIO(record.getSite(), data)
            except:
                print(f"record: trouble with data for {record.getLink()}")
                print(f"record={record}")
                pass

            try:
                # mrun.plateaus(threshold=2.e-3, duration=10, save=args.save, debug=args.debug)
                mrun.plateaus(duration=10, show=False, save=False, debug=args.debug)
            except:
                print(f"record: plateaus detection fails for {record.getLink()}")
                pass

