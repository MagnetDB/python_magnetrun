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
from .. import MRecord

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

import lxml.html as lh
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

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
    url_records = base_url + "site/sba/pages/" + "courbes.php"
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

        """
        since data from url_status are broken
        use E. Verney instead

        # Get data from Status page
        # actually list of site in magnetdb sens
        (_data, jid) = getTable(s, url_status, 2, [1, 3, 4], debug=args.debug)
        if args.debug:
            for item in _data:
                print(f"{item}: status={_data[item]}, jid={jid[item]}")

        print("ordered site data bt time")
        from collections import OrderedDict

        ordered_data = OrderedDict(sorted(_data.items(), key=lambda x: x[1][0]))
        # print(f"ordered_data: {ordered_data}")
        """
        import csv

        _data = {}
        _counter = {}
        print("load site history from M9_M10-history.csv", flush=True)
        with open("M9_M10-history.csv") as f:
            _raw = csv.reader(f)

            for row in _raw:
                print(row)
                try:
                    name = row[1]
                    _magnets = [name]
                    if "??" not in name:
                        status = row[2]
                        housing = row[3]
                        bitter = row[4]
                        
                        tformat = "%Y-%m-%d"
                        created_at = None
                        stopped_at = None

                        if name not in _counter:
                            _counter[name] = 0

                        site = f"{name}_{_counter[name]}"

                        created_at = None
                        stopped_at = None
                        print(f'status={status}, date={row[0]}', flush=True)
                        if status.lower() == "en service":
                            created_at = datetime.datetime.strptime(row[0], tformat)
                            if site in db_Sites:
                                db_Sites[site]["status"] = status.lower()
                                db_Sites[site]["commissioned_at"] = stopped_at
                                db_Sites[site]["housing"] = housing
                                db_Sites[site]["bitter"] = bitter
                            else:
                                db_Sites[site] = {
                                    "name": site,
                                    "description": "",
                                    "status": status.lower(),
                                    "magnets": _magnets,
                                    "records": [],
                                    "commissioned_at": created_at,
                                    "decommissioned_at": stopped_at,
                                    "housing": housing,
                                    "bitter": bitter,
                                }

                        else:
                            stopped_at = datetime.datetime.strptime(row[0], tformat)
                            _counter[name] += 1
                            if site in db_Sites:
                                db_Sites[site]["status"] = status.lower()
                                db_Sites[site]["decommissioned_at"] = stopped_at
                            else:
                                db_Sites[site] = {
                                    "name": site,
                                    "description": "",
                                    "status": status.lower(),
                                    "magnets": _magnets,
                                    "records": [],
                                    "commissioned_at": created_at,
                                    "decommissioned_at": stopped_at,
                                    "housing": housing,
                                    "bitter": bitter,
                                }
                except:
                    print(f'problem loading: {row} -skipped')
                    pass

        print('db_Sites: definition')
        for item, values in db_Sites.items():
            housing = values["housing"]
            name = values["name"]
            if "Bitters" not in item:
                if values['bitter'] == "":
                    values["magnets"].append(f"{housing}Bitters")
                else:
                    values["magnets"].append(values['bitter'])
            values["name"] = f"{housing}_{name}"
            print(f"site={item}: {values}")

        for item in db_Sites:
            del db_Sites[item]['bitter']
        for item, values in db_Sites.items():
            print(f"site={item}: {values}")
        # TODO rename site with housing

        for site, values in db_Sites.items():
            status = values["status"]
            for magnet in values["magnets"]:
                Magnets[magnet] = HMagnet.HMagnet(magnet, "", status, parts=[])
        print("Magnets:")
        for magnet in Magnets:
            print(f"{magnet}: {Magnets[magnet]}")

        Parts = {}
        Confs = {}
        for magnet in Magnets:
            print(f'magnet: {magnet}')
            if "Bitter" not in magnet:
                # if debug:
                print(f"loading helices for: {magnet}", flush=True)
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

        for conf, values in Confs.items():
            print(f"Confs[{conf}]: {values}", flush=True)

        # Get CAD ref for Parts
        PartsCAD = {}
        getPartCADref(s, url_helicescad, PartsCAD, debug=args.debug)
        if debug:
            print("\ngetPartCADref:")
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
            for i, part in Parts[magnet]:
                # print(i, part)
                if part not in PartMagnet:
                    PartMagnet[part] = []
                PartMagnet[part].append(magnet)

        # Create Parts from Magnets
        diameter = {14: 34, 12: 50, 6: 170}
        print(f"\nMagnets ({len(Magnets)}):", flush=True)
        PartName = {}
        db_Magnets = {}
        for magnet in Magnets:
            db_Magnets[magnet] = {
                "name": magnet,
                "status": Magnets[magnet].status,
                "design_office_reference": "",
            }
            for site in db_Sites:
                if magnet in site:
                    if "sites" not in db_Magnets[magnet]:
                        db_Magnets[magnet]["sites"] = []
                    db_Magnets[magnet]["sites"].append(magnet)

            # magconf = Magnets[magnet].MAGfile
            # if magconf:
            #     magconffile = magconf[0]
            #     Carac_Magnets[magnet]['config'] = magconffile

            # TODO read from cvs part <-> geometry (yaml file)
            if "Bitters" not in magnet:
                nhelices = 0
                if Parts[magnet]:
                    db_Magnets[magnet]["parts"] = []
                    for i, part in Parts[magnet]:
                        pname = part
                        db_Magnets[magnet]["parts"].append(pname)
                        if pname not in PartName:
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
                    ] = f"{nhelices} Helices, Phi = {diameter[nhelices]} mm"
                print(
                    f"{magnet}: {db_Magnets[magnet]} - should add {nhelices-1} rings "
                )
            else:
                db_Magnets[magnet]["description"] = "Phi = 400 mm"

        # Create Parts from Materials because in control/monitoring part==mat
        # TODO once Parts is complete no longer necessary
        # ['name', 'description', 'status', 'type', 'design_office_reference', 'material_id'
        print(f"\nMParts ({len(PartsCAD)}):")
        db_Parts = {}
        cad_Parts = {}
        for part in PartsCAD:
            carac = {
                "name": part,
                "description": "",
                "status": "unknown",
                "type": "helix",  # Mats[part].category,
                "design_office_reference": PartsCAD[part][0],
                "material": PartsCAD[part][1],
            }
            # TODO geometry field must be consistant with magnetapi -
            if part in PartName:
                carac["geometry"] = PartName[part][0]
                carac["status"] = PartName[part][1]
                carac["magnets"] = PartName[part][2]
                cad = re.sub("-[a-zA-Z]", "", PartsCAD[part][0])
                if cad in cad_Parts:
                    if part not in cad_Parts[cad]:
                        cad_Parts[cad].append(part)
                else:
                    cad_Parts[cad] = [part]
            print(f"{part}: {carac}")
            db_Parts[part] = carac

        print(f"\ncad/Parts ({len(cad_Parts)}):")


        ordered_data = OrderedDict(sorted(cad_Parts.items(), key=lambda x: x))
        for cad, values in ordered_data.items():
            print(f"{cad} parts={values}")

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
        page = s.get(url=url_records, verify=True)

        housing_names = []
        doc = lh.document_fromstring(page.content)
        # print(f"doc: {lh.tostring(doc)}")
        tr_elements = doc.xpath("//*[@class='example']")
        for i, t in enumerate(tr_elements):
            content = t.text_content().rsplit()  # replace(' dmesg','')
            # print(f"name[{i}]: content={content[0]}")
            housing_names.append(content[0])
        print(f'housing_names: {housing_names}')
        
        record_names = []
        record_timestamps = []
        tformat = "%Y.%m.%d-%H:%M:%S"

        for name in housing_names:
            url_housing = base_url + "/" + name
            print(f'records: {url_housing}', flush=True)
            page = s.get(url=url_housing, verify=True)
            doc = lh.document_fromstring(page.content)
            tr_elements = doc.xpath("//a")  # [@href='example']")
            for i, t in enumerate(tr_elements):
                link = t.get("href")
                # print(f'link={link}', end=": ", flush=True)
                if link.endswith(".txt") and "dmesg" not in link:
                    nlink = ""
                    if link.startswith("./"):
                        nlink = link.replace("./", f"../../../{name}/")
                    else:
                        nlink = f"../../../{name}/{link}"

                    try:
                        record_names.append(nlink)

                        timestamp = (
                            link.split("/")[-1].replace("%20", "").replace(".txt", "")
                        )
                        # print(timestamp)
                        record_timestamps.append(
                            datetime.datetime.strptime(timestamp, tformat)
                        )
                    except:
                        print(f"trouble with record={link}, name={name}, nlink={nlink}, timestamp={timestamp} -record ignored")

                    #print(f'nlink={nlink}', flush=True)

        # Assign records to site from timestamps
        # Create a panda datafram with ['link','timestamp']
        df_records = pd.DataFrame(
            list(zip(record_names, record_timestamps)), columns=["name", "timestamp"]
        )
        df_records.to_csv('df_records.csv')
        
        # for each site
        #     get record with a timestamp in between site.commisionned_at and site.decommisioned_at
        for site, values in db_Sites.items():
            housing = values["housing"]
            t0 = values["commissioned_at"]
            t1 = values["decommissioned_at"]
            print(f'site={site}, housing="{housing}, t0={t0}, t1={t1}', flush=True)
            
            selected_df = None
            if t1 is not None:
                selected = df_records[
                    df_records["timestamp"].between(t0, t1, inclusive="left")
                ]
                print(f"{site}: records={len(selected.index)}")
            else:
                selected = df_records[df_records["timestamp"] >= t0]
                print(f"{site}: records={len(selected.index)} **")

            for link, timestamp in zip(
                selected["name"].tolist(), selected["timestamp"].tolist()
            ):
                if housing in link:
                    record = MRecord.MRecord(timestamp, housing, site, link)
                    values["records"].append(record)

        for site, values in db_Sites.items():
            sname = site.split("_")[0]
            for record in values["records"]:
                if args.check:
                    data = record.getData(s, url_downloads)
                    iodata = StringIO(data)
                    headers = iodata.readline().split()
                    if len(headers) >= 2:
                        insert = headers[1]
                        if not sname.startswith(insert):
                            print(
                                f"{site}: {record} - expected site={sname} got {insert}"
                            )

                    # mrun = MagnetRun.fromStringIO(record.getHousing(), record.getSite(), data)

                    # from ..processing.stats import plateaus
                    # plateaus(Data=mrun.MagnetData, duration=10, save=args.save, debug=args.debug)

                    if args.save:
                        record.saveData(data, args.datadir)

        """
        # Get orphan records
        # How to get all records even those attached to experiments with Bitters only ??
        """
        print("\nOrphaned records:", flush=True)
        record_sites = [db_Sites[site]["records"] for site in db_Sites]
        # print(f"record_names: {record_names[-1]}")
        record_name_sites = [record.getLink() for record in flatten(record_sites)]
        # print(f"record_name_sites: {record_name_sites[-1]}")
        orphan_records = list(
            set(record_names).symmetric_difference(set(flatten(record_name_sites)))
        )

        print(
            f"orphan_records={len(orphan_records)} / {len(record_name_sites)} registered / {len(record_names)} records"
        )
                    
            
        for housing in ["M1", "M3", "M5", "M7", "M8", "M9", "M10"]:
            record_sites = [
                db_Sites[site]["records"]
                for site in db_Sites
                if housing == db_Sites[site]["housing"]
            ]
            record_name_sites = [record.getLink() for record in flatten(record_sites)]
            search_housing = f'/{housing}/'
            record_names_housing = [
                record for record in record_names if search_housing in record
            ]
            orphan_records = list(
                set(record_names_housing).symmetric_difference(
                    set(flatten(record_name_sites))
                )
            )

            print(
                f"{housing}: orphan_records={len(orphan_records)} / {len(record_name_sites)} registered / {len(record_names_housing)} records"
            )

            print(f'{housing}: Saved Orphaned records {len(orphan_records)}', flush=True)
            for orphan in orphan_records:
                # print(f'{orphan} ({type(orphan)})', end="")
                site = "unknown"
                link = orphan 
                timestamp = (
                            link.split("/")[-1].replace("%20", "").replace(".txt", "")
                        )
                orecord = MRecord.MRecord(timestamp, housing, site, link)
                print(orecord, end="")
                data = orecord.getData(s, url_downloads)
                iodata = StringIO(data)
                if args.save:
                    orecord.saveData(data, args.datadir)
                    print('saved', end="")
                print(flush=True)

        # Display site history per site for M9 and M10 only

        print("\nSite History per Housing:", flush=True)

        history = {}
        for housing in housing_names:
            history[housing] = {
                "site": [],
                "commissioned_at": [],
                "decommissioned_at": [],
            }

        for site in db_Sites:
            data = db_Sites[site]
            housing = data["housing"]
            print(f"site={site}, housing={housing}")

            hdata = history[housing]
            hdata["site"].append(site.replace("_", "-"))
            hdata["commissioned_at"].append(data["commissioned_at"])
            hdata["decommissioned_at"].append(data["decommissioned_at"])

        # for housing in ["M1", "M3", "M5", "M7", "M8", "M9", "M10"]:
        for housing in ["M9", "M10"]:
            hdata = history[housing]
            df = pd.DataFrame(hdata)
            ax = df.plot(x="site", y="decommissioned_at", kind="bar")
            df.plot(x="site", y="commissioned_at", kind="bar", ax=ax, color="white")

            today = datetime.date.today()

            tformat = "%Y-%m-%d"
            ymin = df.min(axis=0)["commissioned_at"]
            ymax = today  # datetime.datetime.strptime(today, tformat)
            ax.set_ylim([ymin, ymax])

            ax.get_legend().remove()
            ax.grid(visible=True)

            plt.show()

        # Get orphan part/material
        print(
            "\nOrphaned magnet/part/material - Generate files for import in MagnetDB:", flush=True
        )
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
            values["status"] = "in_stock"

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
        magnet_status = {"en service": "in_operation", "en stock": "in_stock"}
        print("\nGenerate files for import in MagnetDB:")
        for magnet, mvalues in db_Magnets.items():
            if "sites" in mvalues:
                del mvalues["sites"]

            mvalues["status"] = magnet_status[mvalues["status"].lower()]
            mvalues["db_parts"] = []
            if "parts" in mvalues:
                for part in mvalues["parts"]:
                    data_part = db_Parts[part].copy()
                    # print(f"parts[{part}]: {part}, data_part={data_part}")
                    if "magnets" in data_part:
                        del data_part["magnets"]

                    data_part["material_data"] = db_Materials[data_part["material"]]
                    data_part["material"] = data_part.pop("material_data")
                    data_part["status"] = magnet_status[data_part["status"].lower()]
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

        site_status = {"en service": "in_operation", "en stock": "decommisioned"}
        for site, svalues in db_Sites.items():
            housing = svalues["housing"]
            name = svalues["name"]
            # svalues["name"] = f"{housing}_{name}"

            svalues["status"] = site_status[svalues["status"].lower()]
            svalues["commissioned_at"] = str(svalues["commissioned_at"])
            svalues["decommissioned_at"] = str(svalues["decommissioned_at"])
            print(f"db_Sites[{site}]: housing={housing}, magnet={svalues['magnets']}, status={svalues['status']}, commissioned_at={svalues['commissioned_at']}, decommissioned_at={svalues['decommissioned_at']}, records={len(svalues['records'])}", flush=True)

            svalues["data_records"] = []
            for record in svalues["records"]:
                filename = record.getDataFilename()
                # if args.datadir != ".":
                #    filename = f"{args.datadir}/{filename}"
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

            for magnet in svalues["magnets"]:
                print(f"magnets[{site}]: {magnet}", flush=True)

            filename = f'{svalues["name"]}.json'
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                print(f"db_Sites/write_to_json: {filename}", flush=True)
                f.write(json.dumps(svalues, indent=4))


if __name__ == "__main__":
    main()
