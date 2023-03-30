#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import getpass
import sys
import re
import datetime
import requests
import requests.exceptions
from io import StringIO

from .. import HMagnet

from .connect import createSession
from .webscrapping import (
    getTable,
    getSiteRecord,
    getMagnetPart,
    getMaterial,
    getPartCADref,
)
from ..MagnetRun import MagnetRun


def cleanup(remove_site: list, msg: str, site_names: dict, Sites: dict):

    print(f"Remove Site in {remove_site}: {msg}")
    for item in remove_site:
        Sites.pop(item)
        if item in site_names:
            # watch out if site_names is not empty
            if site_names[item]:
                s_ = site_names[item][0]
                site_names[s_] = site_names[item]
                site_names[s_].remove(s_)
            site_names.pop(item)
        else:
            for name in site_names:
                if item in site_names[name]:
                    # print(f'remove {item} from site_names[{name}]')
                    site_names[name].remove(item)
    """
    for name in site_names:
        print(f'site_name[{name}]={site_names[name]}')
    """


def main():
    import argparse
    from .. import python_magnetrun

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="specify user")
    parser.add_argument(
        "--server",
        help="specify server",
        default="https://srv-data-install.lncmi.cnrs.fr/",
    )
    parser.add_argument("--check", help="sanity check for records", action="store_true")
    parser.add_argument("--save", help="save files", action="store_true")
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    args = parser.parse_args()

    if sys.stdin.isatty():
        password = getpass.getpass("Using getpass: ")
    else:
        print("Using readline")
        password = sys.stdin.readline().rstrip()

    if args.save:
        args.check = True

    # print( 'Read: ', password )

    # shall check if host ip up and running
    base_url = args.server
    url_logging = base_url + "site/sba/pages/" + "login.php"
    url_downloads = base_url + "site/sba/pages/" + "courbe.php"
    url_status = base_url + "site/sba/pages/" + "Etat.php"
    url_files = base_url + "site/sba/pages/" + "getfref.php"
    url_helices = base_url + "site/sba/pages/" + "Aimant2.php"
    url_helicescad = base_url + "site/sba/pages/" + "Helice.php"
    url_materials = base_url + "site/sba/pages/" + "Mat.php"
    url_query = (
        base_url + "site/sba/vendor/jqueryFileTree/connectors/jqueryFileTree.php"
    )

    # Fill in your details here to be posted to the login form.
    payload = {"email": args.user, "password": password}

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
        if r.url == url_logging:
            print("check connection failed: Wrong credentials")
            sys.exit(1)

        # Get data from Status page
        # actually list of site in magnetdb sens
        (_data, jid) = getTable(s, url_status, 2, [1, 3, 4], debug=args.debug)
        if args.debug:
            for item in _data:
                print(f"{item}: status={_data[item]}, jid={jid[item]}")

        stopped_at = datetime.datetime.now()
        for item in _data:
            housing = _data[item][2]
            magnet = re.sub("_\d+", "", item)
            status = _data[item][1]  # TODO change status to match magnetdb status
            tformat = "%Y-%m-%d"
            created_at = datetime.datetime.strptime(_data[item][0], tformat)
            stopped_at = datetime.datetime.strptime("2100-01-01", tformat)

            Sites[item] = {
                "name": item,
                "description": "",
                "status": status,
                "magnets": [magnet],
                "records": [],
                "commissioned_at": created_at,
                "decommissioned_at": stopped_at,
            }

        for item in _data:
            res = item.split("_")
            name = res[0]
            if len(res) == 2:
                num = int(res[-1])
                cfg = f"{name}_{num+1}"
            else:
                cfg = f"{name}_1"

            if cfg in _data:
                Sites[item]["decommissioned_at"] = Sites[cfg]["commissioned_at"]

        print("\nSite names:")
        site_names = {}
        for item in _data:
            # grep keys in _data with item_
            match_expr = re.compile(f"{item}_\d+")
            same_cfgs = [key for key in _data if match_expr.match(key) and key != item]
            if same_cfgs:
                site_names[item] = same_cfgs
            if not "_" in item and not same_cfgs:
                site_names[item] = []
        for name in site_names:
            print(f"site_name[{name}]={site_names[name]}")

        # Get records per site
        print("Load records per Site")
        for ID in Sites:  # Mids:
            getSiteRecord(s, url_files, ID, Sites, url_downloads, debug=args.debug)

        # drop site if len(records)==0
        remove_site = []
        for site in Sites:
            if len(Sites[site]["records"]) == 0:
                remove_site.append(site)
        if args.debug:
            print(f"\nSites to be removed (empty list of records): {remove_site}")

        cleanup(remove_site, "empty record list", site_names, Sites)
        for name in site_names:
            print(f"site_name[{name}]={site_names[name]}")

        # Correct status:
        # for site: 'in_study', 'in_operation', 'decommisioned'
        # for magnet: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # for parts: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # En Service -> in_operation
        # En Stock: check latest record time to set status
        # Autre
        for site in Sites:
            if Sites[site]["decommissioned_at"] != stopped_at:
                Sites[site]["status"] = "decommisioned"
            else:
                if Sites[site]["status"].lower() == "en service":
                    Sites[site]["status"] = "in_operation"

                if Sites[site]["status"].lower() == "en stock":
                    latest_record = Sites[site]["records"][-1]
                    latest_time = (
                        latest_record.getTimestamp()
                    )  # not really a timestamp but a datetime.datetime
                    today = (
                        datetime.datetime.now()
                    )  # use today.timestamp() to get timestamp
                    dt = today - latest_time
                    # print(f'{site}: latest record is {dt.days} daysfrom now')
                    if dt.days >= 4:
                        Sites[site]["status"] = "decommisioned"
                        Sites[site]["decommissioned_at"] = latest_time

        if args.debug:
            print(f"\nSites({len(Sites)}): orderer by names")
            for site in site_names:
                housing = Sites[site]["records"][0].getHousing()
                print(
                    f"{Sites[site]['name']}: status={Sites[site]['status']}, housing={housing}, commissioned_at={Sites[site]['commissioned_at']}, decommissioned_at={Sites[site]['decommissioned_at']}, records={len(Sites[site]['records'])}"
                )
                for item in site_names[site]:
                    housing = Sites[item]["records"][0].getHousing()
                    print(
                        f"{Sites[item]['name']}: status={Sites[item]['status']}, housing={housing}, commissioned_at={Sites[item]['commissioned_at']}, decommissioned_at={Sites[item]['decommissioned_at']}, records={len(Sites[item]['records'])}"
                    )

        # if Sites[site]['decommissioned_at'] - Sites[site]['commissioned_at'] < 0 day:
        # transfert record to other site - aka magnet_[n-1] - from name site - aka magnet_n
        # get latest_record from magnet_[n]
        # add update Sites[site]['decommissioned_at'] to latest_time record
        # remove site - aka magnet_[n]
        print("\nCheck sites:")
        remove_site = []
        for site in Sites:
            dt = Sites[site]["decommissioned_at"] - Sites[site]["commissioned_at"]
            if dt.days <= 0:
                print(f"{site}: latest record is {dt.days} days from now")
                res = site.split("_")
                if len(res) != 1:
                    name = res[0]
                    num = int(res[1])
                    previous_site = name
                    if num > 1:
                        previous_site = f"{name}_{num-1}"
                    print(f"{site}: tranfert records to {previous_site}")
                    latest_record = Sites[site]["records"][-1]
                    latest_time = (
                        latest_record.getTimestamp()
                    )  # not really a timestamp but a datetime.datetime
                    Sites[previous_site]["decommissioned_at"] = latest_time
                    for record in Sites[site]["records"]:
                        Sites[previous_site]["records"].append(record)
                        record.setSite(previous_site)
                remove_site.append(site)

        cleanup(remove_site, "less than one day", site_names, Sites)
        for name in site_names:
            print(f"site_name[{name}]={site_names[name]}")

        print("\nCheck sites with same base name:")
        merge_site = {}
        for item in site_names:
            # print(f'site[{item}]: {site_names[item]}')
            num = len(site_names[item])
            merge_site[item] = []
            if num >= 2:
                for i in range(num - 1):
                    if i == 0:
                        dt = (
                            Sites[item]["decommissioned_at"]
                            - Sites[site_names[item][i]]["commissioned_at"]
                        )
                        if dt.days <= 0:
                            merge_site[item].append(item)
                            merge_site[item].append(site_names[item][i])
                    else:
                        dt = (
                            Sites[site_names[item][i]]["decommissioned_at"]
                            - Sites[site_names[item][i + 1]]["commissioned_at"]
                        )
                        if dt.days <= 0:
                            # print(f'{site_names[item][i]} == {site_names[item][i+1]}')
                            merge_site[item].append(site_names[item][i])
                            merge_site[item].append(site_names[item][i + 1])
            # remove duplicate in list
            merge_site[item] = list(dict.fromkeys(merge_site[item]))
        # drop empty key
        merge_site = {i: j for i, j in merge_site.items() if j != []}
        for item in merge_site:
            print(f"{item}: {merge_site[item]}")

        remove_site = []
        for item in merge_site:
            num = len(merge_site[item])
            Sites[merge_site[item][0]]["decommissioned_at"] = Sites[
                merge_site[item][-1]
            ]["decommissioned_at"]
            Sites[merge_site[item][0]]["status"] = Sites[merge_site[item][-1]]["status"]
            for i in range(num):
                # transfert record list to Sites[item][0]
                if i > 0:
                    for record in Sites[merge_site[item][i]]["records"]:
                        Sites[merge_site[item][0]]["records"].append(record)
                        record.setSite(merge_site[item][0])
                    remove_site.append(merge_site[item][i])

        cleanup(remove_site, "merged", site_names, Sites)
        for name in site_names:
            print(f"site_name[{name}]={site_names[name]}")

        # verify if for site in_operation
        # if latest_time - today >= 10 days
        # change status to decommisionning and set decommisionned_at to latest_time
        installation_stop = "12/23/22 00:00:00"
        datetime_stop = datetime.datetime.strptime(
            installation_stop, "%m/%d/%y %H:%M:%S"
        )

        print("\nCheck sites in in_operation:")
        for site in Sites:
            if Sites[site]["status"] == "in_operation":
                latest_record = Sites[site]["records"][-1]
                # print(f'latest_record: {latest_record.housing}')
                latest_time = (
                    latest_record.getTimestamp()
                )  # not really a timestamp but a datetime.datetime
                today = (
                    datetime.datetime.now()
                )  # use today.timestamp() to get timestamp
                dt = today - latest_time
                dt_stop = datetime_stop - latest_time
                if dt.days >= 10:
                    print(
                        f"{site}: latest record {latest_time} on {latest_record.housing} is {dt.days} days from now, {dt_stop.days} days from installation stop {datetime_stop}"
                    )
                    if dt_stop.days >= 7:
                        Sites[site]["status"] = "decommisioned"
                        Sites[site]["decommissioned_at"] = latest_time

        # check records are not shared between sites:
        lst_records = []
        print("\nCheck sites for common records:")
        for site in Sites:
            lst_records.append(
                (set([record.link for record in Sites[site]["records"]]), site)
            )

        for i in range(len(Sites)):
            a_set = lst_records[i][0]
            for j in range(i + 1, len(Sites)):
                b_set = lst_records[j][0]
                if len(a_set.intersection(b_set)) != 0:
                    print(
                        f"records common between {lst_records[i][1]} and {lst_records[j][1]}: {len(a_set.intersection(b_set))}"
                    )

        # Create list of Magnets from sites
        # NB use entries sorted by commisionned date
        # TODO replace Magnets by a dict that is similar to magnetdb magnet
        print(f"\nSites({len(Sites)}):")
        for site in site_names:
            housing = Sites[site]["records"][0].getHousing()
            print(
                f"{Sites[site]['name']}: status={Sites[site]['status']}, housing={housing}, commissioned_at={Sites[site]['commissioned_at']}, decommissioned_at={Sites[site]['decommissioned_at']}, records={len(Sites[site]['records'])}"
            )
            for item in site_names[site]:
                housing = Sites[item]["records"][0].getHousing()
                print(
                    f"{Sites[item]['name']}: status={Sites[item]['status']}, housing={housing}, commissioned_at={Sites[item]['commissioned_at']}, decommissioned_at={Sites[item]['decommissioned_at']}, records={len(Sites[item]['records'])}"
                )

        # print(f"\nMagnets: create and set magnet status")
        for site in site_names:  # sorted(Sites, key=operator.itemgetter(5)):
            magnetID = re.sub("_\d+", "", site)
            # TODO replace HMagnet by a dict that is similar to magnetdb magnet
            # status: inherited from site
            status = Sites[site]["status"]
            if site_names[site]:
                status = Sites[site_names[site][-1]]["status"]
            if status.lower() == "en stock":
                status = "in_stock"
            # print(f'{magnetID}: {status}')
            Magnets[magnetID] = HMagnet.HMagnet(magnetID, "", status, parts=[])
            # print(f'{magnetID}: {Magnets[magnetID]}')

        # print(f"\nMagnets: get parts by magnet and set part status")
        Parts = {}
        Confs = {}
        for magnetID in Magnets:
            magnet = re.sub("_\d+", "", magnetID)
            # print(f"** {magnet}: data={Magnets[magnetID]}")
            if debug:
                print(f"loading helices for: {magnet}")
            getMagnetPart(
                s,
                magnet,
                url_helices,
                magnet,
                Magnets,
                url_materials,
                Parts,
                Mats,
                Confs,
                save=args.save,
                debug=args.debug,
            )
            # print(f"Parts from getMagnetPart[{magnet}] ({len(Parts[magnet])}): {[part for (i,part) in Parts[magnet]]}")

        # Get CAD ref for Parts
        PartsCAD = {}
        getPartCADref(s, url_helicescad, PartsCAD, save=args.save, debug=args.debug)
        print(f"getPartCADref:")
        for key in PartsCAD:
            print(f"{key}: {PartsCAD[key]}")

        #
        # print('Parts: create list of magnet per part')
        PartMagnet = {}
        for magnet in Parts:
            for (i, part) in Parts[magnet]:
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
            magnet = re.sub("_\d+", "", magnetID)
            print(f"** {magnet}: data={Magnets[magnetID]}")
            Carac_Magnets[magnet] = {
                "name": magnet,
                "status": Magnets[magnetID].status,
                "design_office_reference": "",
            }
            if magnet in site_names:
                Carac_Magnets[magnet]["sites"] = [magnet]
                for site in site_names[magnet]:
                    Carac_Magnets[magnet]["sites"].append(site)

            # magconf = Magnets[magnetID].MAGfile
            # if magconf:
            #     magconffile = magconf[0]
            #     Carac_Magnets[magnet]['config'] = magconffile

            if Parts[magnet]:
                Carac_Magnets[magnet]["parts"] = []
                for i, part in Parts[magnet]:
                    print(f"Part[{i}]: {part}")
                    # pname = part.replace('MA','H')
                    pname = part
                    Carac_Magnets[magnet]["parts"].append(pname)
                    if not pname in PartName:
                        latest_magnet = PartMagnet[pname][-1]
                        status = Magnets[latest_magnet].status
                        PartName[pname] = [
                            f"HL-31_H{i+1}",
                            f"{status}",
                            PartMagnet[pname],
                        ]
                        # set status from magnet status??
            print(f"{magnet}: {Carac_Magnets[magnet]}")

        # Create Parts from Materials because in control/monitoring part==mat
        # TODO once Parts is complete no longer necessary
        # ['name', 'description', 'status', 'type', 'design_office_reference', 'material_id'
        print(f"\nMParts ({len(PartsCAD)}):")
        for part in PartsCAD:
            # print(mat, type(Mats[mat]))
            # key = mat.replace('MA','H')
            carac = {
                "name": part,
                "description": "",
                "status": "unknown",
                "type": "helix",  # Mats[part].category,
                "design_office_reference": PartsCAD[part][0],
                "material": PartsCAD[part][1],
            }
            if part in PartName:
                carac["geometry"] = PartName[part][0]
                carac["status"] = PartName[part][1]
                carac["magnets"] = PartName[part][2]
            print(f"{part}: {carac}")

        # TODO replace Materials by a dict that is similar to magnetdb material
        # Ref ou REF???
        getMaterial(s, None, url_materials, Mats, debug=args.debug)
        print(f"\nMaterials ({len(Mats)}):")
        for mat in Mats:
            carac = {
                "name": Mats[mat].name,
                "description": "",
                "t_ref": 293,
                "volumic_mass": 9e3,
                "specific_heat": 0,
                "alpha": 3.6e-3,
                "electrical_conductivity": float(
                    Mats[mat].material["sigma0"].replace(",", ".")
                )
                * 1e6,
                "thermal_conductivity": 380,
                "magnet_permeability": 1,
                "young": 117e9,
                "poisson": 0.33,
                "expansion_coefficient": 18e-6,
                "rpe": float(Mats[mat].material["rpe"]) * 1e6,
            }
            if "nuance" in Mats[mat].material:
                carac["nuance"] = Mats[mat].material["nuance"]
            print(f"{mat}: {carac}")

        # Try to read and make some stats on records
        print("\nRecords:")
        for site in Sites:
            for record in Sites[site]["records"]:
                if args.check:
                    data = record.getData(s, url_downloads)
                    iodata = StringIO(data)
                    headers = iodata.readline().split()
                    if len(headers) >= 2:
                        insert = headers[1]
                        if not site.startswith(insert):
                            print(
                                f"{site}: {record} - expected site={site} got {insert}"
                            )

                    # mrun = MagnetRun.fromStringIO(record.getHousing(), record.getSite(), data)

                    # from ..processing.stats import plateaus
                    # plateaus(Data=mrun.MagnetData, duration=10, save=args.save, debug=args.debug)

                    if args.save:
                        record.saveData(data)

                # break


if __name__ == "__main__":
    main()
