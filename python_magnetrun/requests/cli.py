#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import json
import getpass
import os
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
from ..utils.list import flatten


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
    parser.add_argument("--datadir", help="specify data dir", type=str, default=".")
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    args = parser.parse_args()

    if sys.stdin.isatty():
        password = getpass.getpass("Using getpass: ")
    else:
        print("Using readline")
        password = sys.stdin.readline().rstrip()

    if args.save:
        args.check = True

    if args.datadir != ".":
        if not os.path.exists(args.datadir):
            os.mkdir(args.datadir)

    # print( 'Read: ', password )

    # shall check if host ip up and running
    base_url = args.server
    url_logging = base_url + "site/sba/pages/" + "login.php"
    url_downloads = base_url + "site/sba/pages/" + "courbe.php"
    url_status = base_url + "site/sba/pages/" + "Etat.php"
    url_files = base_url + "site/sba/pages/" + "getfref.php"
    url_helices = base_url + "site/sba/pages/" + "Aimant2.php"
    url_helicescad = base_url + "site/sba/pages/" + "Helice.php"
    url_ringscad = base_url + "site/sba/pages/" + "Bague.php"
    url_materials = base_url + "site/sba/pages/" + "Mat.php"
    url_confs = base_url + "site/sba/pages/downloadM.php"
    url_query = (
        base_url + "site/sba/vendor/jqueryFileTree/connectors/jqueryFileTree.php"
    )

    # Fill in your details here to be posted to the login form.
    payload = {"email": args.user, "password": password}

    # Magnets
    db_Sites = dict()
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
            magnet = re.sub(f"_\\d+", "", item)
            status = _data[item][1]  # TODO change status to match magnetdb status
            tformat = "%Y-%m-%d"
            created_at = datetime.datetime.strptime(_data[item][0], tformat)
            stopped_at = datetime.datetime.strptime("2100-01-01", tformat)

            db_Sites[item] = {
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
                db_Sites[item]["decommissioned_at"] = db_Sites[cfg]["commissioned_at"]

        if args.debug:
            print("\nSite names:")
        site_names = {}
        for item in _data:
            match_expr = re.compile(f"{item}_\\d+")
            same_cfgs = [key for key in _data if match_expr.match(key) and key != item]
            if same_cfgs:
                site_names[item] = same_cfgs
            if not "_" in item and not same_cfgs:
                site_names[item] = []
        if args.debug:
            for name in site_names:
                print(f"site_name[{name}]={site_names[name]}")

        # Get records per site
        print("Load records per Site")
        for ID in db_Sites:  # Mids:
            getSiteRecord(s, url_files, ID, db_Sites, url_downloads, debug=args.debug)

        # drop site if len(records)==0
        remove_site = []
        for site in db_Sites:
            if len(db_Sites[site]["records"]) == 0:
                remove_site.append(site)
        if args.debug:
            print(f"\ndb_Sites to be removed (empty list of records): {remove_site}")

        cleanup(remove_site, "empty record list", site_names, db_Sites)
        if debug:
            for name in site_names:
                print(f"site_name[{name}]={site_names[name]}")

        # Correct status:
        # for site: 'in_study', 'in_operation', 'decommisioned'
        # for magnet: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # for parts: 'in_study', 'in_operation', 'in_stock', 'defunct'
        # En Service -> in_operation
        # En Stock: check latest record time to set status
        # Autre
        for site, values in db_Sites.items():
            if values["decommissioned_at"] != stopped_at:
                values["status"] = "decommisioned"
            else:
                if values["status"].lower() == "en service":
                    values["status"] = "in_operation"

                if values["status"].lower() == "en stock":
                    latest_record = values["records"][-1]
                    latest_time = (
                        latest_record.getTimestamp()
                    )  # not really a timestamp but a datetime.datetime
                    today = (
                        datetime.datetime.now()
                    )  # use today.timestamp() to get timestamp
                    dt = today - latest_time
                    # print(f'{site}: latest record is {dt.days} daysfrom now')
                    if dt.days >= 4:
                        values["status"] = "decommisioned"
                        values["decommissioned_at"] = latest_time

        if args.debug:
            print(f"\ndb_Sites({len(db_Sites)}): orderer by names")
            for site in site_names:
                housing = db_Sites[site]["records"][0].getHousing()
                print(
                    f"{db_Sites[site]['name']}: status={db_Sites[site]['status']}, housing={housing}, commissioned_at={db_Sites[site]['commissioned_at']}, decommissioned_at={db_Sites[site]['decommissioned_at']}, records={len(db_Sites[site]['records'])}"
                )
                for item in site_names[site]:
                    housing = db_Sites[item]["records"][0].getHousing()
                    print(
                        f"{db_Sites[item]['name']}: status={db_Sites[item]['status']}, housing={housing}, commissioned_at={db_Sites[item]['commissioned_at']}, decommissioned_at={db_Sites[item]['decommissioned_at']}, records={len(db_Sites[item]['records'])}"
                    )

        # if db_Sites[site]['decommissioned_at'] - db_Sites[site]['commissioned_at'] < 0 day:
        # transfert record to other site - aka magnet_[n-1] - from name site - aka magnet_n
        # get latest_record from magnet_[n]
        # add update db_Sites[site]['decommissioned_at'] to latest_time record
        # remove site - aka magnet_[n]
        print("\nCheck db_Sites:")
        remove_site = []
        for site, values in db_Sites.items():
            dt = values["decommissioned_at"] - values["commissioned_at"]
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
                    latest_record = values["records"][-1]
                    latest_time = (
                        latest_record.getTimestamp()
                    )  # not really a timestamp but a datetime.datetime
                    db_Sites[previous_site]["decommissioned_at"] = latest_time
                    for record in values["records"]:
                        db_Sites[previous_site]["records"].append(record)
                        record.setSite(previous_site)
                remove_site.append(site)

        cleanup(remove_site, "less than one day", site_names, db_Sites)
        if debug:
            for name in site_names:
                print(f"site_name[{name}]={site_names[name]}")

        print("\nCheck db_Sites with same base name:")
        merge_site = {}
        for item in site_names:
            # print(f'site[{item}]: {site_names[item]}')
            num = len(site_names[item])
            merge_site[item] = []
            if num >= 2:
                for i in range(num - 1):
                    if i == 0:
                        dt = (
                            db_Sites[item]["decommissioned_at"]
                            - db_Sites[site_names[item][i]]["commissioned_at"]
                        )
                        if dt.days <= 0:
                            merge_site[item].append(item)
                            merge_site[item].append(site_names[item][i])
                    else:
                        dt = (
                            db_Sites[site_names[item][i]]["decommissioned_at"]
                            - db_Sites[site_names[item][i + 1]]["commissioned_at"]
                        )
                        if dt.days <= 0:
                            # print(f'{site_names[item][i]} == {site_names[item][i+1]}')
                            merge_site[item].append(site_names[item][i])
                            merge_site[item].append(site_names[item][i + 1])
            # remove duplicate in list
            merge_site[item] = list(dict.fromkeys(merge_site[item]))
        # drop empty key
        merge_site = {i: j for i, j in merge_site.items() if j != []}
        if debug:
            for item in merge_site:
                print(f"{item}: {merge_site[item]}")

        remove_site = []
        for item in merge_site:
            num = len(merge_site[item])
            db_Sites[merge_site[item][0]]["decommissioned_at"] = db_Sites[
                merge_site[item][-1]
            ]["decommissioned_at"]
            db_Sites[merge_site[item][0]]["status"] = db_Sites[merge_site[item][-1]][
                "status"
            ]
            for i in range(num):
                # transfert record list to db_Sites[item][0]
                if i > 0:
                    for record in db_Sites[merge_site[item][i]]["records"]:
                        db_Sites[merge_site[item][0]]["records"].append(record)
                        record.setSite(merge_site[item][0])
                    remove_site.append(merge_site[item][i])

        cleanup(remove_site, "merged", site_names, db_Sites)
        if debug:
            for name in site_names:
                print(f"site_name[{name}]={site_names[name]}")

        # verify if for site in_operation
        # if latest_time - today >= 10 days
        # change status to decommisionning and set decommisionned_at to latest_time
        installation_stop = "12/23/22 00:00:00"
        datetime_stop = datetime.datetime.strptime(
            installation_stop, "%m/%d/%y %H:%M:%S"
        )

        print("\nCheck db_Sites in in_operation:")
        for site, values in db_Sites.items():
            if values["status"] == "in_operation":
                latest_record = values["records"][-1]
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
                        values["status"] = "decommisioned"
                        values["decommissioned_at"] = latest_time

        # check records are not shared between db_Sites:
        lst_records = []
        print("\nCheck db_Sites for common records:")
        for site in db_Sites:
            lst_records.append(
                (set([record.link for record in db_Sites[site]["records"]]), site)
            )

        for i in range(len(db_Sites)):
            a_set = lst_records[i][0]
            for j in range(i + 1, len(db_Sites)):
                b_set = lst_records[j][0]
                if len(a_set.intersection(b_set)) != 0:
                    print(
                        f"records common between {lst_records[i][1]} and {lst_records[j][1]}: {len(a_set.intersection(b_set))}"
                    )

        # Create list of Magnets from db_Sites
        # NB use entries sorted by commisionned date
        # TODO replace Magnets by a dict that is similar to magnetdb magnet
        print(f"\ndb_Sites({len(db_Sites)}):")
        for site in site_names:
            values = db_Sites[site]
            housing = values["records"][0].getHousing()
            print(
                f"{values['name']}: status={values['status']}, housing={housing}, commissioned_at={values['commissioned_at']}, decommissioned_at={values['decommissioned_at']}, records={len(values['records'])}"
            )
            for item in site_names[site]:
                svalues = db_Sites[item]
                housing = svalues["records"][0].getHousing()
                print(
                    f"{svalues['name']}: status={svalues['status']}, housing={housing}, commissioned_at={svalues['commissioned_at']}, decommissioned_at={svalues['decommissioned_at']}, records={len(svalues['records'])}"
                )

        # print(f"\nMagnets: create and set magnet status")
        for site in site_names:  # sorted(db_Sites, key=operator.itemgetter(5)):
            magnetID = re.sub("_\\d+", "", site)
            status = db_Sites[site]["status"]
            if site_names[site]:
                status = db_Sites[site_names[site][-1]]["status"]
            if status.lower() == "en stock":
                status = "in_stock"
            Magnets[magnetID] = HMagnet.HMagnet(magnetID, "", status, parts=[])

        Parts = {}
        Confs = {}
        for magnetID in Magnets:
            magnet = re.sub("_\\d+", "", magnetID)
            if debug:
                print(f"loading helices for: {magnet}")
            getMagnetPart(
                s,
                magnet,
                url_helices,
                Magnets,
                url_materials,
                Parts,
                Mats,
                url_confs,
                Confs,
                datadir=args.datadir,
                save=args.save,
                debug=args.debug,
            )
            """
            print(
                f"{magnet}: Magnets={Magnets}, Mats={Mats}, Parts={Parts}, Confs={Confs[magnet]}"
            )
            """

        for conf, values in Confs.items():
            print(f"Confs[{conf}]: {values}")

        # Get CAD ref for Parts
        PartsCAD = {}
        getPartCADref(s, url_helicescad, PartsCAD, debug=args.debug)
        if debug:
            print(f"\ngetPartCADref:")
            for key in PartsCAD:
                print(f"{key}: {PartsCAD[key]}")

        """
        # Try to get rings like Helices - not working
        getPartCADref(s, url_ringscad, PartsCAD, params={"REF": ""}, debug=True)
        if debug:
            print(f"\ngetPartCADref:")
            for key in PartsCAD:
                print(f"{key}: {PartsCAD[key]}")
        """

        PartMagnet = {}
        for magnet in Parts:
            for (i, part) in Parts[magnet]:
                # print(i, part)
                if not part in PartMagnet:
                    PartMagnet[part] = []
                PartMagnet[part].append(magnet)

        # Create Parts from Magnets
        print(f"\nMagnets ({len(Magnets)}):")
        PartName = {}
        db_Magnets = {}
        for magnetID in Magnets:
            magnet = re.sub("_\\d+", "", magnetID)
            db_Magnets[magnet] = {
                "name": magnet,
                "status": Magnets[magnetID].status,
                "design_office_reference": "",
            }
            if magnet in site_names:
                db_Magnets[magnet]["sites"] = [magnet]
                for site in site_names[magnet]:
                    db_Magnets[magnet]["sites"].append(site)

            # magconf = Magnets[magnetID].MAGfile
            # if magconf:
            #     magconffile = magconf[0]
            #     Carac_Magnets[magnet]['config'] = magconffile

            nhelices = 0
            if Parts[magnet]:
                db_Magnets[magnet]["parts"] = []
                for i, part in Parts[magnet]:
                    pname = part
                    db_Magnets[magnet]["parts"].append(pname)
                    if not pname in PartName:
                        latest_magnet = PartMagnet[pname][-1]
                        status = Magnets[latest_magnet].status
                        PartName[pname] = [
                            f"HL-31_H{i+1}",
                            f"{status}",
                            PartMagnet[pname],
                        ]

                nhelices = len(db_Magnets[magnet]["parts"])
                db_Magnets[magnet][
                    "description"
                ] = f'{nhelices} Helices, Phi = xx mm'
            print(f"{magnet}: {db_Magnets[magnet]} - should add {nhelices-1} rings ")

        # Create Parts from Materials because in control/monitoring part==mat
        # TODO once Parts is complete no longer necessary
        # ['name', 'description', 'status', 'type', 'design_office_reference', 'material_id'
        print(f"\nMParts ({len(PartsCAD)}):")
        db_Parts = {}
        for part in PartsCAD:
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
            db_Parts[part] = carac

        getMaterial(s, None, url_materials, Mats, debug=args.debug)
        print(f"\nMaterials ({len(Mats)}):")
        db_Materials = {}
        for mat in Mats:
            carac = {
                "name": Mats[mat].name,
                "description": "",
                "t_ref": 293,
                "volumic_mass": 9e3,
                "specific_heat": 385,
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
            db_Materials[Mats[mat].name] = carac

        # Try to read and make some stats on records
        print("\nRecords:")
        for site, values in db_Sites.items():
            for record in values["records"]:
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
                        record.saveData(data, args.datadir)

        # Get orphan part/material
        print("\nOrphaned magnet/part/material - Generate files for import in MagnetDB:")
        magnet_names = [db_Magnets[magnet]["name"] for magnet in db_Magnets]
        site_magnets = [
            svalues["magnets"]
            for site, svalues in db_Sites.items()
            if "magnets" in svalues
        ]
        orphan_magnets = list(
            set(magnet_names).symmetric_difference(set(flatten(site_magnets)))
        )

        part_names = [db_Parts[part]["name"] for part in db_Parts]
        part_magnets = [
            mvalues["parts"]
            for magnet, mvalues in db_Magnets.items()
            if "parts" in mvalues
        ]
        orphan_parts = list(
            set(part_names).symmetric_difference(set(flatten(part_magnets)))
        )

        material_names = [mvalues["name"] for material, mvalues in db_Materials.items()]
        part_materials = [pvalues["material"] for part, pvalues in db_Parts.items()]
        orphan_materials = list(
            set(material_names).symmetric_difference(set(part_materials))
        )

        # print(f"orphan_materials={orphan_materials}")
        for mat in orphan_materials:
            values = db_Materials[mat]
            filename = f'{values["name"]}.json'
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                print(f"Orphan_Materials/write_to_json: {filename}")
                f.write(json.dumps(values, indent=4))

        # print(f"orphan_parts={orphan_parts}")
        for part in orphan_parts:
            values = db_Parts[part]

            values["material_data"] = db_Materials[values["material"]].copy()
            values["material"] = values.pop("material_data")

            filename = f'{values["name"]}.json'
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                print(f"Orphan_Parts/write_to_json: {filename}")
                f.write(json.dumps(values, indent=4))

        print(f"orphan_magnets={orphan_magnets}")
        for magnet in orphan_magnets:
            values = db_Magnets[magnet]

        # For MagnetDB
        print("\nGenerate files for import in MagnetDB:")
        for magnet, mvalues in db_Magnets.items():
            if "sites" in mvalues:
                del mvalues["sites"]

            mvalues["db_parts"] = []
            if "parts" in mvalues:
                for part in mvalues["parts"]:
                    data_part = db_Parts[part].copy()
                    # print(f"parts[{part}]: {part}, data_part={data_part}")
                    if "magnets" in data_part:
                        del data_part["magnets"]

                    data_part["material_data"] = db_Materials[data_part["material"]]
                    data_part["material"] = data_part.pop("material_data")

                    mvalues["db_parts"].append(data_part)

                del mvalues["parts"]
            else:
                print(f"db_Magnets[{magnet}]: {mvalues} - no parts")

            mvalues["parts"] = mvalues["db_parts"]
            del mvalues["db_parts"]

            filename = f'{mvalues["name"]}.json'
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                print(f"db_Magnets/write_to_json: {filename}")
                f.write(json.dumps(mvalues, indent=4))

        for site, svalues in db_Sites.items():
            housing = svalues["records"][0].getHousing()
            name = svalues["name"]
            svalues["name"] = f"{housing}_{name}"
            print(f"db_Sites[{site}]: housing={housing}, magnet={svalues['magnets']}")

            svalues["commissioned_at"] = str(svalues["commissioned_at"])
            svalues["decommissioned_at"] = str(svalues["decommissioned_at"])

            svalues["data_records"] = []
            for record in svalues["records"]:
                filename = record.getDataFilename()
                if args.datadir != ".":
                    filename = f"{args.datadir}/{filename}"
                data_record = {
                    "name": record.getDataFilename(),
                    "description": "",
                    "file": filename,
                    "site": svalues["name"],
                }
                svalues["data_records"].append(data_record)

            del svalues["records"]
            svalues["records"] = svalues["data_records"]
            del svalues["data_records"]

            if housing in ["M8", "M9", "M10"]:
                svalues['magnets'].append(f'{housing}Bitters')
            for magnet in svalues['magnets']:
                print(f'magnets[{site}]: {magnet}')

            filename = f'{svalues["name"]}.json'
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                print(f"db_Sites/write_to_json: {filename}")
                f.write(json.dumps(svalues, indent=4))


if __name__ == "__main__":
    main()
