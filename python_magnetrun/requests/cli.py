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

from .. import MRecord
from .. import GObject
from .. import HMagnet

from .connect import createSession, download
from .webscrapping import createSession, download, getTable, getSiteRecord, getMagnetPart, getMaterial

def main():
    import argparse
    from .. import python_magnetrun
    import matplotlib
    import matplotlib.pyplot as plt
    
    requests.packages.urllib3.disable_warnings()
    
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
    Sites = dict()
    SiteRecords = dict()
    Magnets = dict()
    Mats = dict()
    debug = args.debug

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = createSession(s, url_logging, payload, args.debug)
        # print('connect:', p)
        # test connection
        r = s.get(url=url_status, verify=True)
        if r.url == url_logging:
            print("check connection failed: Wrong credentials" )
            sys.exit(1)
        
        # Get data from Status page
        # actually list of site in magnetdb sens
        (_data, jid) = getTable(s, url_status, 2, [1,3,4], debug=args.debug)
        # for item in _data:
        #     print(f'{item}: status={_data[item]}, jid={jid[item]}')
        
        for item in _data:
            # print(f'{item}: status={_data[item]}, jid={jid[item]}')
            housing = _data[item][2]
            magnet = re.sub('_\d+','',item)
            status = _data[item][1]
            tformat="%Y-%m-%d"
            created_at = datetime.datetime.strptime(_data[item][0], tformat)
            stopped_at = datetime.datetime.strptime("2100-01-01", tformat)

            Sites[item] = {'name': item,
                           'description': '',
                           'status': status,
                           'magnets': [magnet],
                           'records': [],
                           'commissioned_at': created_at,
                           'decommissioned_at': stopped_at}

        for item in _data:
            # print(f'{item}: status={_data[item]}, jid={jid[item]}')

            # grep keys in _data with item_
            match_expr = re.compile(f'{item}_\d+')
            same_cfgs = [ key for key in _data if match_expr.match(key) and key != item]
            if same_cfgs:
                # print('same site cfg:', same_cfgs)
                Sites[item]['decommissioned_at'] = Sites[same_cfgs[0]]['commissioned_at']
                same_cfgs.pop(0)

        for site in Sites:
            print(f'site: {site}={Sites[site]}')

        # Get records per site
        for ID in Sites: #Mids:
            getSiteRecord(s, url_files, ID, Sites, url_downloads, debug=args.debug)
            print(f"getSiteRecord({ID}): records={len(Sites[ID]['records'])}")

        # Create list of Magnets from sites
        for site in Sites:
            magnetID = re.sub('_\d+','', site)
            Magnets[magnetID] = HMagnet.HMagnet(magnetID, 0, None, "Unknown", 0)

        
        Parts = {}
        for magnetID in Magnets:
            magnet = re.sub('_\d+','',magnetID)
            print(f"** {magnet}: data={Magnets[magnetID]}")
            if debug:
                print(f"loading helices for: {magnet}")
            getMagnetPart(s, magnet, url_helices, magnet, Magnets, url_materials, Parts, Mats, save=args.save, debug=args.debug)
        if debug:
            print("\nMagnets: ")
        print("\nMagnets:", len(Magnets))

        PartName = {}
        Carac_Magnets = {}
        for magnetID in Magnets:
            # print(magnet, type(Magnets[magnet]))
            magnet = re.sub('_\d+','',magnetID)
            print(f"** {magnet}: data={Magnets[magnetID]}")
            Carac_Magnets[magnet] = {'name':magnet,'status':Magnets[magnetID].status}
            magconf = Magnets[magnetID].MAGfile
            if magconf:
                magconffile = magconf[0]
                Carac_Magnets[magnet]['config'] = magconffile
                if Parts[magnet]:
                    Carac_Magnets[magnet]['parts'] = []
                    for part in Parts[magnet]:
                        pname = part[-1].replace('MA','H')
                        pid = part[0]
                        Carac_Magnets[magnet]['parts'].append(pname)
                        if not pname in PartName:
                            PartName[pname] = f"HL-31_H{pid}"
            print(f"name: {magnet}, carac={Carac_Magnets[magnet]}")
        
        print("\nMaterials: already found:", len(Mats))
        # Ref ou REF???
        getMaterial(s, None, url_materials, Mats, debug=args.debug)

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

    #     sites = {}
    #     for magnet in Magnets:
    #         housing_records={}
    #         if magnet in MagnetRecords:
    #             housing_records={}
    #             for record in MagnetRecords[magnet]:
    #                 filename = record.link.replace('../../../','')
    #                 filename = filename.replace('/','_').replace('%20','-')
    #                 housing = filename.split('_')[0]
    #                 if housing in housing_records:
    #                     housing_records[housing].append(filename)
    #                 else:
    #                     housing_records[housing] = [filename]

    #             sites[ f'{housing}_{magnet}' ] = housing_records[housing]

    #     print("\nSites:")
    #     for site in sites:
    #         housing = site.split('_')[0]
    #         carac = {'name': site, 'status': 'in_study'}
    #         print(site, carac)
            
    #     print("\nRecords:")
    #     for site in sites:
    #         for file in sites[site]:
    #             print(site, {'file':file,'site':site})

    # print("\nSum up: ")
    # print("\nMagnets:")
    # for magnet in Magnets:
    #     if not magnet in MagnetRecords:
    #         MagnetRecords[magnet] = []
    #     nhelix = 0
    #     if magnet in Carac_Magnets:
    #         if 'parts' in Carac_Magnets[magnet]:
    #             nhelix = len(Carac_Magnets[magnet]['parts'])
        
    #     print(f"** {magnet}: status={Magnets[magnet].getStatus()}, records={len(MagnetRecords[magnet])}, helices={nhelix}" )

    # # print("\nMagnets in Operation:")
    # # for magnet in Magnets:
    # #     ## Broken to json:
    # #     #try:
    # #     if Magnets[magnet].getStatus() == "En service":
    # #         if args.save:
    # #             fo = open(magnet + ".json", "w", newline='\n')
    # #             fo.write(Magnets[magnet].to_json())
    # #             fo.close()
    # #         for record in MagnetRecords[magnet]:
    # #             print(f"magnet={magnet}, record={record.getSite()}, link={record.getLink()},url_downloads={url_downloads}")
    # #             data = record.getData(s, url_downloads, save=args.save)
    # #             try:
    # #                 mrun = python_magnetrun.MagnetRun.fromStringIO(record.getSite(), data)
    # #             except:
    # #                 print(f"record: trouble with data for {record.getLink()}")
    # #                 print(f"record={record}")
    # #                 pass

    # #             try:
    # #                 # mrun.plateaus(threshold=2.e-3, duration=10, save=args.save, debug=args.debug)
    # #                 mrun.plateaus(duration=10, save=args.save, debug=args.debug)
    # #             except:
    # #                 print(f"record: plateaus detection fails for {record.getLink()}")
    # #                 pass

    # # print("\nMaterials:")
    # # for mat in Mats:
    # #     print(mat, ":", Mats[mat])
    # #     if args.save:
    # #         fo = open(mat + ".json", "w", newline='\n')
    # #         fo.write(Mats[mat].to_json())
    # #         fo.close()

if __name__ == "__main__":
    main()
