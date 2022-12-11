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
from .webscrapping import createSession, download, getTable, getMagnetRecord, getMagnetPart

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
    Magnets = dict()
    MagnetRecords = dict()
    MagnetComps = dict()
    Status = dict()
    Mats = dict()
    debug = args.debug

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = createSession(s, url_logging, payload, args.debug)
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
            getMagnetRecord(s, url_files, magnetID, Magnets, url_downloads, MagnetRecords, save=args.save, debug=args.debug)
            # print(f'getMagnetRecord({magnetID}): records={len(MagnetRecords[magnetID])}')
            Magnets[magnetID].setStatus(Status[magnetID][-1])
            Magnets[magnetID].setIndex(jid[magnetID])

        Parts = {}
        for magnet in Magnets:
            print(f"** {magnet}: status={Magnets[magnet].getStatus()}")
            if debug:
                print(f"loading helices for: {magnet}")

            getMagnetPart(s, magnet, url_helices, magnetID, Magnets, url_materials, Parts, Mats, save=args.save, debug=args.debug)
        if debug:
            print("\nMagnets: ")
        print("\nMagnets:", len(Magnets))

        PartName = {}
        for magnet in Magnets:
            # print(magnet, type(Magnets[magnet]))
            carac = {'name':magnet.replace('M','M'),'status':Magnets[magnet].status}
            magconf = Magnets[magnet].MAGfile
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
            print(magnet.replace('M','M'), carac)
        
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
            
        print("\nRecords:")
        for site in sites:
            for file in sites[site]:
                print(site, {'file':file,'site':site})

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

    # print("\nMagnets in Operation:")
    # for magnet in Magnets:
    #     ## Broken to json:
    #     #try:
    #     if Magnets[magnet].getStatus() == "En service":
    #         if args.save:
    #             fo = open(magnet + ".json", "w", newline='\n')
    #             fo.write(Magnets[magnet].to_json())
    #             fo.close()
    #         for record in MagnetRecords[magnet]:
    #             print(f"magnet={magnet}, record={record.getSite()}, link={record.getLink()},url_downloads={url_downloads}")
    #             data = record.getData(s, url_downloads, save=args.save)
    #             try:
    #                 mrun = python_magnetrun.MagnetRun.fromStringIO(record.getSite(), data)
    #             except:
    #                 print(f"record: trouble with data for {record.getLink()}")
    #                 print(f"record={record}")
    #                 pass

    #             try:
    #                 # mrun.plateaus(threshold=2.e-3, duration=10, save=args.save, debug=args.debug)
    #                 mrun.plateaus(duration=10, save=args.save, debug=args.debug)
    #             except:
    #                 print(f"record: plateaus detection fails for {record.getLink()}")
    #                 pass

    # print("\nMaterials:")
    # for mat in Mats:
    #     print(mat, ":", Mats[mat])
    #     if args.save:
    #         fo = open(mat + ".json", "w", newline='\n')
    #         fo.write(Mats[mat].to_json())
    #         fo.close()

if __name__ == "__main__":
    main()
