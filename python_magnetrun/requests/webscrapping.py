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
from .. import MRecord
from .. import GObject

from .connect import createSession, download

# for M1:
# table fileTreeDemo_1, ul, <li  class="file ext_txt">, <a href=.., rel="filename" /a> </li>

def getTable(session, url_data, index, indices, delimiter='//tbody', param=None, debug=False):
    """
    get table data from url_data
    
    index:
    indices:
    """

    # Perform some webscrapping to get all table data
    # see https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
    if param is None:
        page = session.get(url=url_data, verify=True)
    else:
        page = session.get(url=url_data, params=param, verify=True)
    if page.status_code != 200 :
        print("cannot logging to %s" % url_data)
        sys.exit(1)
    page.raise_for_status()
    if debug:
        print( "connect:", page.url, page.status_code )
        print( "index:", index )
        print( "indices:", indices )
        print( "params:", param )
        # print(f"page.text={page.text} **")

    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)

    # from php source
    # table : id datatable, tr id=row, thead, tbody, <td class="sorting_1"...>
    # Parse data that are stored between <tbody>..</tbody> of HTML
    tr_elements = doc.xpath(delimiter) # '//tbody')

    #Create empty list ## Better to have a dict??
    jid = None
    Mid = None
    Mjid = dict()
    Mdata = dict()
    Found = dict()
    
    if not tr_elements:
        return (Mdata, Mjid)

    if debug:
        print(f"detected tables[delimiter={delimiter}]: {tr_elements}")
        for i,t in enumerate(tr_elements[0]):
            print(f"{i}: content={t.text_content()}, type={type(t.text_content())}")
            for j,d in enumerate(t):
                print(f'\t{j}:{d.text_content()}')

    #For each row, store each first element (header) and an empty list
    for i,t in enumerate(tr_elements[0]):
        name=t.text_content()
        # get date ID status comment from sub element
        data = []
        for j,d in enumerate(t):
            jname = d.text_content()
            if debug:
                print(f'\t{j}:{name}')
            if j+1 == index:
                if param:
                    jid = jname[jname.find("(")+1:jname.find(")")]
                    # print("jid=", jid)
                Mid = re.sub(' (.*)','',jname)
            if j+1 in indices:
                data.append(jname)
        # shall check wether key is already defined for sanity
        if Mid == "-" :
            print(f"{name} index: no entry")
        else:
            if Mid in Mdata:
                Found[Mid] = Found[Mid] + 1
                Mid = f'{Mid}_{Found[Mid]}'
            else:
                Found[Mid] = 0
            Mdata[Mid] = data
            Mjid[Mid] = jid

    # Mids = sorted(set(Mids)) #uniq only: list(set(Mids))
    if debug:
        print(f"Data found: Mdata={Mdata}, jid={Mjid}")
    return (Mdata, Mjid)

def getMaterial(session, materialID: int, url_materials, Mats: dict, debug=False):
    """get material"""
    #print(f'getMaterial({materialID})')

    if materialID is None:
        r = session.post(url_materials, data={ 'compact:': 'on', 'formsubmit': 'OK' }, verify=True)
        r.raise_for_status()
        # print("post Material: ", r.url, r.status_code)
        html = lh.fromstring(r.text.encode(r.encoding))
        sigmas = html.xpath('//input[@name="CONDUCTIVITE"]/@value')
        elasticlimits = html.xpath('//input[@name="LE"]/@value')
        refs = html.xpath('//input[@name="REF"]/@value')
        nuances = html.xpath('//input[@name="NUANCE"]/@value')
        if debug:
            if len(Mats.keys()) != len(refs)-1:
                print("Materials in main list:", len(refs)-1)

        for i,ref in enumerate(refs):
            # ref is lxml.etree._ElementUnicodeResult
            if not ref in Mats and not "finir" in ref:
                if debug:
                    print("ref:", ref, type(ref), sigmas[i], elasticlimits[i])
                Mats[ref] = GObject.GObject(str(ref), 0,0,
                                            {"sigma0":str(sigmas[i]), "rpe":str(elasticlimits[i]), "nuance":str(nuances[i])},
                                            "helix", "Unknown")
    else:
        r = session.post(url_materials, data={ 'REF': materialID, 'compact:': 'on', 'formsubmit': 'OK' }, verify=True)
        r.raise_for_status()
        html = lh.fromstring(r.text.encode(r.encoding))
        conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
        elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
        nuance = html.xpath('//input[@name="NUANCE"]/@value')[-1]
        # print(materialID, nuance)
        if not materialID in Mats:
            Mats[materialID] = GObject.GObject(materialID, 0,0,
                                               {"sigma0":str(conductivity), "rpe": str(elasticlimit), "nuance": nuance},
                                               "helix", "Unknown")
                
                


def getMagnetPart(session, magnet, url_helices, magnetID, Magnets, url_materials, Parts, Mats, save=False, debug=False):
    """get parts for a given magnet"""
    #print(f'getMagnetPart({magnet})')

    params_helix = (
        ('ref', magnet),
    )

    hindices = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,19]
    res = getTable(session, url_helices, 1, hindices, param=params_helix, debug=debug)
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
        # print(f"{data}:")
        for i in range(len(helices[data])-1):
            materialID = re.sub('H.* / ','',helices[data][i])
            if debug:
                print("%s:" % materialID )
            if materialID != '-':
                Parts[magnet].append([i,materialID.replace('H','MA')])
                getMaterial(session, materialID, url_materials, Mats, debug)
                #Magnets[magnet].addGObject(materialID)
                # # MagnetComps ???
                # if not magnet in MagnetComps:
                #     MagnetComps[magnet] = []
                # MagnetComps[magnet].append(materialID)
                # if debug:
                #     print("MagnetComps[%s].append(%s)" % (magnet,materialID) )
                
                #     print("Material: %s" % materialID,
                #           "Conductivity=", conductivity,
                #           "ElasticLimit=", elasticlimit)

        # # Shall be attached to site in magnetdb sense
        # MAGconf = helices[data][-1]
        # MAGconf.replace('  \t\t\t\t\t\t','')
        # MAGconf.replace('\n',',')
        # Magconf_files = MAGconf.split(' ')
        # Magconf_files = [f for f in Magconf_files if f.endswith('.conf')]
        # if debug:
        #     print(f"MAGconfile={Magconf_files}  **")
        # Magnets[magnet].setMAGfile(Magconf_files)

    
def getSiteRecord(session, url_data, ID, Sites, url_downloads, debug=False):
    """get records for a given ID"""
    #print(f'getSiteRecord({ID})')
    
    # To get files for ID
    params_links = (
        ('ref', re.sub('_\d+','',ID)),
        ('link', ''),
    )

    r = session.get(url=url_data, params=params_links, verify=True)
    if debug:
        print( f"data: url={r.url}, status={r.status_code}, encoding={r.encoding}, text={r.text}")
    if r.status_code != 200:
        print("error %d loading %s" % (p.status_code, url_data) )
        sys.exit(1)
    r.raise_for_status()

    # Get list of records attached to ID
    for f in r.text.split('<br>'):
        if f and not '~' in f :
            replace_str='<a href='+'\''+url_downloads+'?file='

            data = f.replace(replace_str,'').replace('</a>','') .replace('\'>',': ').split(': ')
            link = data[0].replace(' ','%20')
            link = re.sub('<a?(.*?)file=', '', link,  flags=re.DOTALL)
            site = link.replace('../../../','')
            site = re.sub('/.*txt','',site)

            housing = link.replace('../../../','').split('/')[0]
            
            tformat="%Y.%m.%d - %H:%M:%S"
            timestamp = datetime.datetime.strptime(data[1].replace('.txt',''), tformat)
            # print(f'timestamp: {timestamp}, {type(timestamp)}')

            # # Download a specific file
            # params_downloads = 'file=%s&download=1' % link
            # html = download(session, url_downloads, param=params_downloads, link=link)

            # lines = html.split('\n')[0] # get 1st line
            # lines_items = lines.split('\t')

            # actual_id = None
            # if len(lines_items) >= 2:
            #     actual_id = lines_items[1]
            # if debug:
            #     print(f'{magnetID}: actual_id={actual_id}, site={site} link={link}, param={params_downloads}')
            
            # if not actual_id:
            #     if debug: print("%s: no name defined for Magnet" % link)

            #else:
            record = MRecord.MRecord(timestamp, housing, ID, link)
            created_at = Sites[ID]['commissioned_at']
            stopped_at = Sites[ID]['decommissioned_at']
            if record.timestamp < created_at or record.timestamp > stopped_at:
                if debug: print(f'{ID}: {record} dropped')
            else:
                Sites[ID]['records'].append(record)
            if debug:
                print(f'{ID}: site={site} link={link}')
            # data = record.getData(session, url_downloads, save)

            # if actual_id != magnetID:
            #     print(f"record: incoherent data magnetID {magnetID} actual_id: {actual_id} - {timestamp}, {site} {link}" )
            #     # TO change magnetID in txt once downloaded
            #     data = data.replace(actual_id,magnetID)
            #     # overwrite data
            #     if save:
            #         filename = link.replace('../../../','')
            #         filename = filename.replace('/','_').replace('%20','-')
            #         # print("save to %s" % filename)
            #         fo = open(filename, "w", newline='\n')
            #         fo.write(data)
            #         fo.close()

        else:
            if debug:
                print(f'getSiteRecords({ID}): f={f}')

