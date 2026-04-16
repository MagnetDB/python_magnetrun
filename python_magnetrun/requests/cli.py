#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import datetime
import getpass
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import lxml.html as lh
import matplotlib.pyplot as plt
import pandas as pd
import requests
import requests.exceptions
from natsort import natsorted

from ..log_utils import setup_logging
from ..utils.list import flatten
from ..utils.timestamps import parse_filename_timestamp
from .connect import createSession
from .HMagnet import HMagnet
from .MRecord import MRecord
from .webscrapping import (
    getCirrusFiles,
    getMagnetPart,
    getMaterial,
    getPartCADref,
    getRingCADref,
)

# Setup logger
logger = logging.getLogger(__name__)

_BASE_CAD_RE = re.compile(r"^([A-Z]+-\d+-\d+)", re.IGNORECASE)


def normalize_cad_ref(value: str) -> str:
    """Strip suffix from a CAD reference, returning only the base (e.g. 'HL-34-020').

    Examples:
        'HL-34-020-A'   -> 'HL-34-020'
        'HL-34-020A'    -> 'HL-34-020'
        'HL-34-020MC'   -> 'HL-34-020'
        'HL-34-020.brep'-> 'HL-34-020'
    """
    stem = Path(value).stem
    m = _BASE_CAD_RE.match(stem)
    return m.group(1).upper() if m else stem


def cleanup(remove_site: list, msg: str, site_names: dict, Sites: dict):
    logger.info(f"Remove Site in {remove_site}: {msg}")
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
                    logger.debug(f"remove {item} from site_names[{name}]")
                    site_names[name].remove(item)


def main():
    import argparse

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
    parser.add_argument(
        "--load-cirrus",
        help="load logs and XMLs from cirrus.php",
        action="store_true",
    )
    parser.add_argument(
        "--cirrus-feed",
        help="specify cirrus feed (A1, A2, A3, A4, etc.)",
        type=str,
        default="A1",
    )
    parser.add_argument(
        "--list-parts",
        help="list all parts (helices and/or rings) from srv-data",
        action="store_true",
    )
    parser.add_argument(
        "--part-type",
        help="filter parts by type when using --list-parts",
        type=str,
        choices=["helix", "ring", "all"],
        default="all",
    )
    parser.add_argument(
        "--log-level",
        help="set logging level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log-file",
        help="save log output to a file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--geometry-csv",
        help="CSV file with two columns 'geometry' and 'yamfile' to override geometry field",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level, log_file=args.log_file if args.log_file else None)
    logger.setLevel(log_level)

    # Infer debug mode from log level
    debug = log_level == logging.DEBUG

    logger.debug(f"args: {args}")
    logger.debug(f"debug mode: {debug}")

    if sys.stdin.isatty():
        password = getpass.getpass("Using getpass: ")
    else:
        logger.info("Using readline")
        password = sys.stdin.readline().rstrip()

    if args.datadir != "." and not os.path.exists(args.datadir):
        os.mkdir(args.datadir)

    logger.debug(f"Read password: {'***' if password else 'empty'}")

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
    url_cirrus = base_url + "site/sba/pages/" + "cirrus.php"
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

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = createSession(s, url_logging, payload, debug)
        # test connection
        r = s.get(url=url_status, verify=True)
        if r.url == url_logging:
            logger.error("check connection failed: Wrong credentials")
            sys.exit(1)

        # Load cirrus logs and XMLs if requested
        if args.load_cirrus:
            logger.info(f"Loading cirrus files for feed: {args.cirrus_feed}")
            cirrus_data = getCirrusFiles(
                s,
                url_cirrus,
                feed=args.cirrus_feed,
                datadir=args.datadir,
                save=args.save,
                debug=debug,
            )
            logger.info(
                f"Loaded {len(cirrus_data['logs'])} log files and {len(cirrus_data['xmls'])} XML files"
            )
            for log in cirrus_data["logs"]:
                logger.info(f"  Log: {log['name']} - {log['url']}")
            for xml in cirrus_data["xmls"]:
                logger.info(f"  XML: {xml['name']} - {xml['url']}")

            # If only loading cirrus files, exit here
            if args.load_cirrus and not args.check:
                logger.info("Cirrus files loaded successfully.")
                return

        """
        since data from url_status are broken
        use E. Verney instead

        # Get data from Status page
        # actually list of site in magnetdb sens
        (_data, jid) = getTable(s, url_status, 2, [1, 3, 4], debug=args.debug)
        if args.debug:
            for item in _data:
                logger.debug(f"{item}: status={_data[item]}, jid={jid[item]}")

        logger.info("ordered site data bt time")
        from collections import OrderedDict

        ordered_data = OrderedDict(sorted(_data.items(), key=lambda x: x[1][0]))
        logger.debug(f"ordered_data: {ordered_data}")
        """
        import csv

        _data = {}
        _counter = {}
        logger.info("Load site history from M9_M10-history.csv")
        with open("M9_M10-history.csv") as f:
            _raw = csv.reader(f)

            for row in _raw:
                logger.debug(row)
                try:
                    name = row[1]
                    _magnets = [name]
                    if "??" not in name:
                        status = row[2] if row[2] != "HS" else "En stock"
                        housing = row[3] if row[3] != "HS" else ""
                        bitter = row[4]

                        tformat = "%Y-%m-%d"
                        created_at = None
                        stopped_at = None

                        if name not in _counter:
                            _counter[name] = 0

                        # see L680-689
                        site = f"{name}_{_counter[name]}"

                        created_at = None
                        stopped_at = None
                        logger.debug(f"status={status}, date={row[0]}")
                        if status.lower() == "en service":
                            created_at = datetime.datetime.strptime(row[0], tformat)
                            # rename site with create_at # see L680-689
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
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"problem loading: {row} - {str(e)} - skipped")
                    pass

        logger.info("db_Sites: definition")
        for item, values in db_Sites.items():
            housing = values["housing"]
            name = values["name"]
            if "Bitters" not in item:
                if values["bitter"] == "":
                    values["magnets"].append(f"{housing}Bitters")
                else:
                    values["magnets"].append(values["bitter"])
            # et ici? # see L680-689
            values["name"] = f"{housing}_{name}"
            logger.debug(f"site={item}: {values}")

        for item in db_Sites:
            del db_Sites[item]["bitter"]
        for item, values in db_Sites.items():
            logger.debug(f"site={item}: {values}")
        # TODO rename site with housing
        logger.info(f" {len(db_Sites)} sites loaded")

        for _site, values in db_Sites.items():
            status = values["status"]
            for magnet in values["magnets"]:
                Magnets[magnet] = HMagnet(magnet, "", status, parts=[])

        logger.info(f"Magnets: {len(Magnets)} loaded")
        for magnet in Magnets:
            logger.debug(f"{magnet}: {Magnets[magnet]}")

        Parts = {}
        Confs = {}
        for magnet in Magnets:
            logger.debug(f"magnet: {magnet}")
            if "Bitter" not in magnet:
                logger.info(f"loading helices for: {magnet}")
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
                    debug=debug,
                )

        for conf, values in Confs.items():
            logger.debug(f"Confs[{conf}]: {values}")

        # Get CAD ref for Parts
        PartsCAD = {}
        logger.info("Loading CAD references for helices from Helice.php")
        getPartCADref(s, url_helicescad, PartsCAD, part_type="helix", debug=debug)
        if debug:
            logger.debug("\ngetPartCADref (Helices):")
            for key in PartsCAD:
                logger.debug(f"{key}: {PartsCAD[key]}")
        logger.info(f"Done; {len(PartsCAD)} CAD references loaded")

        # Get CAD ref for Rings from Bague.php
        logger.info("Loading CAD references for rings from Bague.php")
        getRingCADref(s, url_ringscad, PartsCAD, debug=debug)
        if debug:
            logger.debug("\ngetRingCADref (Rings):")
            for key in PartsCAD:
                if len(PartsCAD[key]) > 3 and PartsCAD[key][3] == "ring":
                    logger.debug(f"{key}: {PartsCAD[key]}")
        rings_count = sum(1 for v in PartsCAD.values() if v and v[-1] == "ring")
        logger.info(f"Done; {rings_count} CAD references loaded")

        # List parts if requested
        if args.list_parts:
            logger.info("\n" + "=" * 80)
            logger.info(f"Parts from srv-data server: {args.server}")
            logger.info("=" * 80)

            # Separate parts by type
            helices = {
                k: v for k, v in PartsCAD.items() if len(v) > 3 and v[3] == "helix"
            }
            rings = {k: v for k, v in PartsCAD.items() if len(v) > 3 and v[3] == "ring"}

            # Display based on filter
            if args.part_type in ["helix", "all"]:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"HELICES ({len(helices)} found)")
                logger.info(f"{'=' * 80}")
                logger.info(
                    f"{'Name':<20} {'CAD Ref':<20} {'Material':<15} {'Geometry':<20}"
                )
                logger.info("-" * 80)
                for name, data in sorted(helices.items()):
                    cad_ref = data[0] if len(data) > 0 else "N/A"
                    material = data[1] if len(data) > 1 else "N/A"
                    # Geometry is derived from CAD Ref by removing suffix letter
                    # Remove 2 chars if ends with "-X" (e.g., "HL-27-031-C" → "HL-27-031")
                    # Remove 1 char if ends with "X" only (e.g., "HL-27-034C" → "HL-27-034")
                    if re.search(r"-[A-Za-z]$", data[2]):
                        geometry = cad_ref[:-2]  # Remove "-X"
                    elif re.search(r"[A-Za-z]$", data[2]):
                        geometry = data[2][:-1]  # Remove "X"
                    else:
                        geometry = (
                            data[2] if len(data) > 2 else "N/A"
                        )  # No suffix letter
                    logger.info(
                        f"{name:<20} {cad_ref:<20} {material:<15} {geometry:<20}"
                    )

            if args.part_type in ["ring", "all"]:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"RINGS ({len(rings)} found)")
                logger.info(f"{'=' * 80}")
                logger.info(
                    f"{'Name':<20} {'CAD Ref':<20} {'Material':<15} {'Geometry':<20}"
                )
                logger.info("-" * 80)
                for name, data in sorted(rings.items()):
                    cad_ref = data[0] if len(data) > 0 else "N/A"
                    material = data[1] if len(data) > 1 else "N/A"
                    if re.search(r"-[A-Za-z]$", data[2]):
                        geometry = cad_ref[:-2]  # Remove "-X"
                    elif re.search(r"[A-Za-z]$", data[2]):
                        geometry = data[2][:-1]  # Remove "X"
                    else:
                        geometry = (
                            data[2] if len(data) > 2 else "N/A"
                        )  # No suffix letter
                    logger.info(
                        f"{name:<20} {cad_ref:<20} {material:<15} {geometry:<20}"
                    )

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Total: {len(helices)} helices, {len(rings)} rings")
            logger.info(f"{'=' * 80}\n")

            # Exit after listing if no other operations requested
            if not args.check and not args.save:
                sys.exit(0)

        PartMagnet = {}
        for magnet in Parts:
            for i, part in Parts[magnet]:
                logger.debug(f"{i}, {part}")
                if part not in PartMagnet:
                    PartMagnet[part] = []
                PartMagnet[part].append(magnet)

        # Create Parts from Magnets
        diameter = {14: 34, 12: 50, 6: 170}
        logger.info(f"Magnets ({len(Magnets)}): create and attached to site")
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
                                f"HL-31_H{i + 1}",
                                f"{status}",
                                PartMagnet[pname],
                            ]

                    nhelices = len(db_Magnets[magnet]["parts"])
                    db_Magnets[magnet][
                        "description"
                    ] = f"{nhelices} Helices, Phi = {diameter[nhelices]} mm"
                logger.info(
                    f"{magnet}: {db_Magnets[magnet]} - should add {nhelices - 1} rings "
                )
            else:
                db_Magnets[magnet]["description"] = "Phi = 400 mm"
        logger.info("Done")

        # Create Parts from Materials because in control/monitoring part==mat
        # TODO once Parts is complete no longer necessary
        # ['name', 'description', 'status', 'type', 'design_office_reference', 'material_id'
        logger.info(f"MParts ({len(PartsCAD)}):")
        db_Parts = {}
        cad_Parts = {}
        for part in PartsCAD:
            # PartsCAD[part] = [cad_ref, material, geometry, part_type]
            part_type = PartsCAD[part][3] if len(PartsCAD[part]) > 3 else "helix"
            logger.info(f" {part} ({part_type}) -- {part in PartName}")
            carac = {
                "name": part,
                "description": "",
                "status": "unknown",
                "type": part_type,  # Use the part_type from PartsCAD (helix or ring)
                "design_office_reference": PartsCAD[part][0],
                "material": PartsCAD[part][1],
                "geometry": normalize_cad_ref(PartsCAD[part][0]),
            }
            # TODO geometry field must be consistant with magnetapi -
            if part in PartName:
                carac["status"] = PartName[part][1]
                carac["magnets"] = PartName[part][2]
                cad = normalize_cad_ref(PartsCAD[part][0])
                carac["geometry"] = cad
                if cad in cad_Parts:
                    if part not in cad_Parts[cad]:
                        cad_Parts[cad].append(part)
                else:
                    cad_Parts[cad] = [part]
            logger.debug(f"{part}: {carac}")
            db_Parts[part] = carac
        logger.info("Done")

        if args.geometry_csv:
            logger.info(f"Applying geometry overrides from {args.geometry_csv}")
            import csv

            geometry_override = {}
            with open(args.geometry_csv, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    geometry_override[row["cad_value"]] = row["file"]
            logger.info(
                f"Loaded {len(geometry_override)} geometry overrides from {args.geometry_csv}"
            )
            for part, carac in db_Parts.items():
                geom = carac.get("geometry")
                if geom in geometry_override:
                    logger.debug(
                        f"{part}: geometry overridden {geom} -> {geometry_override[geom]}"
                    )
                    carac["geometry"] = geometry_override[geom]

        ordered_data = OrderedDict(sorted(cad_Parts.items(), key=lambda x: x))
        for cad, values in ordered_data.items():
            logger.debug(f"{cad} parts={values}")
        logger.info(f"cad/Parts ({len(cad_Parts)}): Done")

        getMaterial(s, None, url_materials, Mats, debug=debug)
        logger.info(f"Materials ({len(Mats)}): definitions")
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
            logger.debug(f"{mat}: {carac}")
            db_Materials[Mats[mat].name] = carac
        logger.info("Done")

        # Try to read and make some stats on records
        logger.info("Records:")
        page = s.get(url=url_records, verify=True)

        housing_names = []
        doc = lh.document_fromstring(page.content)
        logger.debug(f"doc: {lh.tostring(doc)}")
        tr_elements = doc.xpath("//*[@class='example']")
        for i, t in enumerate(tr_elements):
            content = t.text_content().rsplit()  # replace(' dmesg','')
            logger.debug(f"name[{i}]: content={content[0]}")
            housing_names.append(content[0])
        logger.debug(f"housing_names: {housing_names}")

        record_names = []
        record_timestamps = []
        tformat = "%Y.%m.%d-%H:%M:%S"

        for name in housing_names:
            url_housing = base_url + "/" + name
            page = s.get(url=url_housing, verify=True)
            doc = lh.document_fromstring(page.content)
            tr_elements = doc.xpath("//a")  # [@href='example']")
            for _i, t in enumerate(tr_elements):
                link = t.get("href")
                logger.debug(f"link={link}")
                if link.endswith(".txt") and "dmesg" not in link:
                    nlink = ""
                    if link.startswith("./"):
                        nlink = link.replace("./", f"../../../{name}/")
                    else:
                        nlink = f"../../../{name}/{link}"

                    timestamp: str = ""
                    try:
                        timestamp = (
                            link.split("/")[-1].replace("%20", "").replace(".txt", "")
                        )
                        # print(timestamp)
                        parsed_ts = datetime.datetime.strptime(timestamp, tformat)
                        record_names.append(nlink)
                        record_timestamps.append(parsed_ts)
                    except (ValueError, OSError) as e:
                        logger.warning(
                            f"trouble with record={link}, name={name}, nlink={nlink}, timestamp={timestamp} -record ignored: {e}"
                        )

                    logger.debug(f"nlink={nlink}")
            logger.info(f"records: {url_housing} - found {len(record_names)} records")
        logger.info(f" {len(record_names)} records found")

        # Assign records to site from timestamps
        # Create a panda datafram with ['link','timestamp']
        df_records = pd.DataFrame(
            list(zip(record_names, record_timestamps, strict=False)),
            columns=["name", "timestamp"],
        )
        df_records.to_csv("df_records.csv")

        # for each site
        #     get record with a timestamp in between site.commisionned_at and site.decommisioned_at
        logger.info("Assign records to sites based on timestamps:")
        for site, values in db_Sites.items():
            housing = values["housing"]
            t0 = values["commissioned_at"]
            t1 = values["decommissioned_at"]

            # Skip sites with invalid timestamps
            if t0 is None:
                logger.warning(f"{site}: skipping - commissioned_at is None")
                continue

            selected = None
            if t1 is not None:
                selected = df_records[
                    df_records["timestamp"].between(t0, t1, inclusive="left")
                ]
                # logger.info(f"{site}: records={len(selected.index)}")
            else:
                selected = df_records[df_records["timestamp"] >= t0]
            #  logger.info(f"{site}: records={len(selected.index)} **")

            for link, timestamp in zip(
                selected["name"].tolist(), selected["timestamp"].tolist(), strict=False
            ):
                if housing in link:
                    record = MRecord(timestamp, housing, site, link)
                    values["records"].append(record)
            first_record = (
                values["records"][0].getDataFilename()
                if len(values["records"]) > 0
                else "N/A"
            )
            last_record = (
                values["records"][-1].getDataFilename()
                if len(values["records"]) > 0
                else "N/A"
            )
            logger.info(
                f'site={site}, housing="{housing}, t0={t0}, t1={t1}, assigned records={len(values["records"])} from {first_record} to {last_record}'
            )
        logger.info("Assign records to sites based on timestamps Done")

        logger.info("Save records and Check if requested:")
        for site, values in db_Sites.items():
            sname = site.split("_")[0]
            for record in values["records"]:
                if args.save:
                    filepath = os.path.join(args.datadir, record.getDataFilename())
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        logger.info(
                            f"File {filepath} already exists, skipping download"
                        )
                        continue
                    data = record.getData(s, url_downloads)
                    if args.check:
                        iodata = StringIO(data)
                        headers = iodata.readline().split()
                        if len(headers) >= 2:
                            insert = headers[1]
                            if not sname.startswith(insert):
                                logger.warning(
                                    f"{site}: {record} - expected site={sname} got {insert}"
                                )

                    # mrun = MagnetRun.fromStringIO(record.getHousing(), record.getSite(), data)

                    # from ..processing.stats import plateaus
                    # plateaus(Data=mrun.MagnetData, duration=10, save=args.save, debug=args.debug)

                    if not len(data) > 0:
                        raise ValueError(
                            f"Cannot save record {record} for site {site} to {args.datadir}, filename={record.getDataFilename()}: empty file"
                        )
                    record.saveData(data, args.datadir)
                    logger.info(
                        f"Saved record {record} for site {site} to {args.datadir}, filename={record.getDataFilename()}"
                    )
        logger.info("Save records and Check if requested Done")

        """
        # Get orphan records
        # How to get all records even those attached to experiments with Bitters only ??
        """
        if args.check:
            logger.info("\nOrphaned records:")
            record_sites = [db_Sites[site]["records"] for site in db_Sites]
            # print(f"record_names: {record_names[-1]}")
            record_name_sites = [record.getLink() for record in flatten(record_sites)]
            logger.debug(f"record_name_sites: {record_name_sites[-1]}")
            orphan_records = list(
                set(record_names).symmetric_difference(set(flatten(record_name_sites)))
            )

            logger.info(
                f"orphan_records={len(orphan_records)} / {len(record_name_sites)} registered / {len(record_names)} records"
            )

            for housing in ["M1", "M3", "M5", "M7", "M8", "M9", "M10"]:
                record_sites = [
                    db_Sites[site]["records"]
                    for site in db_Sites
                    if housing == db_Sites[site]["housing"]
                ]
                record_name_sites = [
                    record.getLink() for record in flatten(record_sites)
                ]
                search_housing = f"/{housing}/"
                record_names_housing = [
                    record for record in record_names if search_housing in record
                ]
                orphan_records = list(
                    set(record_names_housing).symmetric_difference(
                        set(flatten(record_name_sites))
                    )
                )

                logger.info(
                    f"{housing}: orphan_records={len(orphan_records)} / {len(record_name_sites)} registered / {len(record_names_housing)} records"
                )

                logger.info(f"{housing}: Saved Orphaned records {len(orphan_records)}")
                for orphan in orphan_records:
                    logger.debug(f"{orphan} ({type(orphan)})")
                    site = "unknown"
                    link = orphan
                    fname = link.split("/")[-1].replace("%20", " ").strip()
                    timestamp = parse_filename_timestamp(fname)
                    logger.info(f"orphan record: {orphan}, timestamp={timestamp}")
                    orecord = MRecord(timestamp, housing, site, link)
                    logger.debug(f"{orecord}")
                    filepath = os.path.join(args.datadir, orecord.getDataFilename())
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        logger.debug(
                            f"File {filepath} already exists, skipping download"
                        )
                        continue
                    data = orecord.getData(s, url_downloads)
                    iodata = StringIO(data)
                    if args.save:
                        orecord.saveData(data, args.datadir)

        # Display site history per site for M9 and M10 only

        logger.info("\nSite History per Housing:")

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
            logger.debug(f"site={site}, housing={housing}")

            logger.debug(f"site={site}, housing={housing}, data={data}")
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
        logger.info(
            "\nOrphaned magnet/part/material - Generate files for import in MagnetDB:"
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

        logger.debug(f"orphan_materials={orphan_materials}")
        for mat in orphan_materials:
            values = db_Materials[mat]
            filename = f"{values['name']}.json"
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                logger.info(f"Orphan_Materials/write_to_json: {filename}")
                f.write(json.dumps(values, indent=4))

        logger.debug(f"orphan_parts={orphan_parts}")
        for part in orphan_parts:
            values = db_Parts[part]

            values["material_data"] = db_Materials[values["material"]].copy()
            values["material"] = values.pop("material_data")
            values["status"] = "in_stock"
            if values["geometry"] != "":
                filename = f"{values['name']}.json"
                if args.datadir != ".":
                    filename = f"{args.datadir}/{filename}"
                with open(filename, "w") as f:
                    logger.info(f"Orphan_Parts/write_to_json: {filename}")
                    f.write(json.dumps(values, indent=4))

        logger.debug(f"orphan_magnets={orphan_magnets}")
        for magnet in orphan_magnets:
            values = db_Magnets[magnet]

        # For MagnetDB
        magnet_status = {"en service": "in_operation", "en stock": "in_stock"}
        magnet_json_files = []
        logger.info("\nGenerate files for import in MagnetDB:")
        for magnet, mvalues in db_Magnets.items():
            if "sites" in mvalues:
                del mvalues["sites"]

            mvalues["status"] = magnet_status[mvalues["status"].lower()]
            mvalues["db_parts"] = []
            if "parts" in mvalues:
                for part in mvalues["parts"]:
                    data_part = db_Parts[part].copy()
                    logger.debug(f"parts[{part}]: {part}, data_part={data_part}")
                    if "magnets" in data_part:
                        del data_part["magnets"]

                    data_part["material_data"] = db_Materials[data_part["material"]]
                    data_part["material"] = data_part.pop("material_data")
                    data_part["status"] = magnet_status[data_part["status"].lower()]
                    mvalues["db_parts"].append(data_part)

                del mvalues["parts"]
            else:
                logger.warning(f"db_Magnets[{magnet}]: {mvalues} - no parts")

            mvalues["parts"] = mvalues["db_parts"]
            del mvalues["db_parts"]

            filename = f"{mvalues['name']}.json"
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                logger.info(f"db_Magnets/write_to_json: {filename}")
                f.write(json.dumps(mvalues, indent=4))
            magnet_json_files.append(filename)

        logger.info("\nCreated magnet JSON files:")
        for f in natsorted(magnet_json_files):
            logger.info(f"  {f}")

        site_status = {"en service": "in_operation", "en stock": "decommisioned"}
        site_json_files = []

        logger.info("\nGenerate files for import in MagnetDB:")
        for site, svalues in sorted(
            db_Sites.items(),
            key=lambda x: (x[1]["commissioned_at"] is None, x[1]["commissioned_at"]),
        ):
            housing = svalues["housing"]
            name = svalues["name"]
            # svalues["name"] = f"{housing}_{name}"

            svalues["status"] = site_status[svalues["status"].lower()]
            svalues["commissioned_at"] = str(svalues["commissioned_at"])
            svalues["decommissioned_at"] = str(svalues["decommissioned_at"])
            logger.info(
                f"db_Sites[{site}]: housing={housing}, magnet={svalues['magnets']}, status={svalues['status']}, commissioned_at={svalues['commissioned_at']}, decommissioned_at={svalues['decommissioned_at']}, records={len(svalues['records'])}"
            )

            svalues["data_records"] = []
            for record in svalues["records"]:
                filename = record.getDataFilename()
                # if args.datadir != ".":
                #    filename = f"{args.datadir}/{filename}"
                data_record = {
                    "name": record.getDataFilename(),
                    "description": "",
                    "file": filename,
                }
                svalues["data_records"].append(data_record)

            del svalues["records"]
            svalues["records"] = svalues["data_records"]
            del svalues["data_records"]

            for magnet in svalues["magnets"]:
                logger.debug(f"magnets[{site}]: {magnet}")

            # filename = f'{svalues["name"]}.json'
            _commissioned_str = svalues["commissioned_at"]
            try:
                _dt = datetime.datetime.fromisoformat(_commissioned_str)
                _date_str = _dt.strftime("%Y%m%d")
            except (ValueError, TypeError):
                _date_str = _commissioned_str

            # filename = f'{housing}_{_date_str}.json'
            filename = f"{svalues['name']}.json"
            if args.datadir != ".":
                filename = f"{args.datadir}/{filename}"
            with open(filename, "w") as f:
                logger.info(f"db_Sites/write_to_json: {filename}")
                f.write(json.dumps(svalues, indent=4))
            site_json_files.append(filename)

        logger.info("\nCreated site JSON files:")
        for f in natsorted(site_json_files):
            logger.info(f"  {f}")
        logger.info("\nDone.")


if __name__ == "__main__":
    main()
