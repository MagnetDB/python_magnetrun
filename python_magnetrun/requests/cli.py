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

from .. import HMagnet

from .connect import createSession
from .webscrapping import getTable, getSiteRecord, getMagnetPart, getMaterial
from ..MagnetRun import MagnetRun

def main():
    import argparse
    from .. import python_magnetrun
    import matplotlib
    import matplotlib.pyplot as plt
    
    requests.packages.urllib3.disable_warnings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="specify user")
    parser.add_argument("--server", help="specify server", default="https://srv-data-install.lncmi.cnrs.fr/")
    parser.add_argument("--check", help="sanity check for records", action='store_true')
    parser.add_argument("--save", help="save files", action='store_true')
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
    args = parser.parse_args()

    if sys.stdin.isatty():
        password = getpass.getpass('Using getpass: ')
    else:
        print( 'Using readline' )
        password = sys.stdin.readline().rstrip()

    if args.save:
        args.check = True

    # print( 'Read: ', password )

    # shall check if host ip up and running
    base_url=args.server
    url_logging=base_url + "site/sba/pages/" + "login.php"
    url_downloads=base_url + "site/sba/pages/" + "courbe.php"
    url_status=base_url + "site/sba/pages/" + "Etat.php"
    url_files=base_url + "site/sba/pages/" + "getfref.php"
    url_helices=base_url + "site/sba/pages/" + "Aimant2.php"
    url_materials=base_url + "site/sba/pages/" + "Mat.php"
    url_query=base_url + "site/sba/vendor/jqueryFileTree/connectors/jqueryFileTree.php"


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
        # test connection
        r = s.get(url=url_status, verify=True)
        # print('try to connect:', r.url)
        # print('should be:', url_logging)
        # print('return url correct=', (r.url == url_logging))
        if r.url == url_logging:
            print("check connection failed: Wrong credentials" )
            sys.exit(1)
        
        # Get data from Status page
        # actually list of site in magnetdb sens
        # eventually get also commentaire - if Démonté in commentaire then magnet status=defunct
        (_data, jid) = getTable(s, url_status, 2, [1,3,4], debug=args.debug)
        # for item in _data:
        #     print(f'{item}: status={_data[item]}, jid={jid[item]}')
        
        for item in _data:
            # print(f'{item}: status={_data[item]}, jid={jid[item]}')
            housing = _data[item][2]
            magnet = re.sub('_\d+','',item)
            status = _data[item][1] # TODO change status to match magnetdb status
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
            res = item.split('_')
            name = res[0]
            if len(res) == 2:
                num = int(res[-1])
                cfg = f'{name}_{num+1}'
            else:
                cfg = f'{name}_1'
                # print(f'{item} -> {cfg}')
                
            if cfg in _data:
                Sites[item]['decommissioned_at'] = Sites[cfg]['commissioned_at']
        # for site in Sites:
        #     print(f'site: {site}={Sites[site]}')

        print('\nSite names:')
        site_names = {}
        for item in _data:
            # print(f'{item}: status={_data[item]}, jid={jid[item]}')

            # grep keys in _data with item_
            match_expr = re.compile(f'{item}_\d+')
            same_cfgs = [ key for key in _data if match_expr.match(key) and key != item]
            if same_cfgs:
                # print('same site cfg:', same_cfgs)
                site_names[item] = same_cfgs
            if not '_' in item and not same_cfgs:
                site_names[item] = []

        for item in site_names:
            print(f'{item}: {site_names[item]}')

        # Get records per site
        for ID in Sites: #Mids:
            getSiteRecord(s, url_files, ID, Sites, url_downloads, debug=args.debug)
            # print(f"getSiteRecord({ID}): records={len(Sites[ID]['records'])}")

        # Correct status:
        # for site: 'in_study', 'in_operation', 'decommisioned'
        # for magnet: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # for parts: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # En Service -> in_operation
        # En Stock
        # Autre
        for site in Sites:
            if Sites[site]['decommissioned_at'] != stopped_at:
                Sites[site]['status'] = 'decommisioned'
            else:
                if Sites[site]['status'].lower() == 'en service':
                    Sites[site]['status'] = 'in_operation'

            if Sites[site]['records']:
                housing = Sites[site]['records'][0].getHousing()
                # housing = Sites[site]['records'][-1]
                # print(f"{Sites[site]['name']}: status={Sites[site]['status']}, housing={housing}, commissioned_at={Sites[site]['commissioned_at']}, decommissioned_at={Sites[site]['decommissioned_at']}")

        # Create list of Magnets from sites
        # NB use entries sorted by commisionned date
        # TODO replace Magnets by a dict that is similar to magnetdb magnet
        print(f"\nSites({len(Sites)}): orderer by names")
        for site in site_names:
            print(f"{Sites[site]['name']}: status={Sites[site]['status']}, housing={housing}, commissioned_at={Sites[site]['commissioned_at']}, decommissioned_at={Sites[site]['decommissioned_at']}, records={len(Sites[site]['records'])}")
            for item in site_names[site]:
                print(f"{Sites[item]['name']}: status={Sites[item]['status']}, housing={housing}, commissioned_at={Sites[item]['commissioned_at']}, decommissioned_at={Sites[item]['decommissioned_at']}, records={len(Sites[item]['records'])}")

        # import operator
        # print(f"\nSites({len(Sites)}): orderer by commisioned_at")
        # for site in sorted(Sites, key=operator.itemgetter(5)):
        #     print(f"{Sites[site]['name']}: status={Sites[site]['status']}, housing={housing}, commissioned_at={Sites[site]['commissioned_at']}, decommissioned_at={Sites[site]['decommissioned_at']}, records={len(Sites[site]['records'])}")
        #     # print(f"{Sites[site]['name']}: {Sites[site]}")
            
        # print(f"\nMagnets: create and set magnet status")
        for site in site_names: #sorted(Sites, key=operator.itemgetter(5)):
            magnetID = re.sub('_\d+','', site)
            # TODO replace HMagnet by a dict that is similar to magnetdb magnet
            # status: inherited from site
            status = Sites[site]['status']
            if site_names[site]:
                status = Sites[site_names[site][-1]]['status']
            if status.lower() == 'en stock':
                status = 'in_stock'
            # print(f'{magnetID}: {status}')
            Magnets[magnetID] = HMagnet.HMagnet(magnetID, None, status, parts=[])
            # print(f'{magnetID}: {Magnets[magnetID]}')
        
        # print(f"\nMagnets: get parts by magnet and set part status")
        Parts = {}
        for magnetID in Magnets:
            magnet = re.sub('_\d+','',magnetID)
            # print(f"** {magnet}: data={Magnets[magnetID]}")
            if debug:
                print(f"loading helices for: {magnet}")
            getMagnetPart(s, magnet, url_helices, magnet, Magnets, url_materials, Parts, Mats, save=args.save, debug=args.debug)
            # print(f"Parts from getMagnetPart[{magnet}] ({len(Parts[magnet])}): {[part for (i,part) in Parts[magnet]]}")

        # for magnetID in Magnets:
        #     print(f'{magnetID}: {Magnets[magnetID]}')
            
        
        #
        # print('Parts: create list of magnet per part')
        PartMagnet = {}
        for magnet in Parts:
            for (i,part) in Parts[magnet]:
                # print(i, part)
                if not part in PartMagnet:
                    PartMagnet[part] = []
                PartMagnet[part].append(magnet)
        # for part in PartMagnet:
        #     print(f'{part} : magnets={PartMagnet[part]}')
            
        # Create Parts from Magnets
        # TODO replace Parts by a dict that is similar to magnetdb part
        # 'name', 'description', 'status', 'design_office_reference'
        print(f"\nMagnets ({len(Magnets)}):")
        PartName = {}
        Carac_Magnets = {}
        for magnetID in Magnets:
            # print(magnet, type(Magnets[magnet]))
            magnet = re.sub('_\d+','',magnetID)
            # print(f"** {magnet}: data={Magnets[magnetID]}")
            Carac_Magnets[magnet] = {'name':magnet, 'status':Magnets[magnetID].status, 'design_office_reference': ''}
            if magnet in site_names:
                Carac_Magnets[magnet]['sites'] = [magnet]
                for site in site_names[magnet]:
                    Carac_Magnets[magnet]['sites'].append(site)
                    
            # magconf = Magnets[magnetID].MAGfile
            # if magconf:
            #     magconffile = magconf[0]
            #     Carac_Magnets[magnet]['config'] = magconffile
            
            if Parts[magnet]:
                Carac_Magnets[magnet]['parts'] = []
                for i,part in Parts[magnet]:
                    pname = part.replace('MA','H')
                    Carac_Magnets[magnet]['parts'].append(pname)
                    if not pname in PartName:
                        latest_magnet = PartMagnet[pname][-1]
                        status = Magnets[latest_magnet].status
                        PartName[pname] = [f"HL-31_H{i}", f"{status}", PartMagnet[pname]]
                        # set status from magnet status??
            print(f"{magnet}: {Carac_Magnets[magnet]}")
        

        # Create Parts from Materials because in control/monitoring part==mat
        # TODO once Parts is complete no longer necessary
        # ['name', 'description', 'status', 'type', 'design_office_reference', 'material_id'
        print(f"\nMParts ({len(Mats)}):")
        for mat in Mats:
            # print(mat, type(Mats[mat]))
            key = mat.replace('MA','H')
            carac = {'name': mat.replace('MA','H'),
                     'description': '',
                     'status': 'unknown',
                     'type': Mats[mat].category,
                     'design_office_reference': '',
                     'material': mat
                    }
            if key in PartName:
                carac['geometry'] = PartName[key][0]
                carac['status'] = PartName[key][1]
                carac['magets'] = PartName[key][2]
            print(f"{key}: {carac}")
            
        # TODO replace Materials by a dict that is similar to magnetdb material
        # Ref ou REF???
        getMaterial(s, None, url_materials, Mats, debug=args.debug)
        print(f"\nMaterials ({len(Mats)}):")
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
            print(f"{mat}: {carac}")

        # Try to read and make some stats on records
        print('\nRecords:')
        for site in Sites:
            for record in Sites[site]['records']:
                print(f'{site}: {record}')

                if args.check:
                    data = record.getData(s, url_downloads)
                    try:
                        mrun = MagnetRun.fromStringIO(record.getHousing(), record.getSite(), data)
                    except:
                        print(f"record: trouble with data for {record.getLink()}")
                        print(f"record={record}")

                    # from ..processing.stats import plateaus
                    # plateaus(Data=mrun.MagnetData, duration=10, save=args.save, debug=args.debug)

                    if args.save:
                        record.saveData(data)

                # break

                

if __name__ == "__main__":
    main()
