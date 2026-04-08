#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import datetime
import logging
import re
import sys
from typing import Any

import lxml.html as lh

# import jsonpickle
from .connect import download
from .GObject import GObject
from .MRecord import MRecord

logger = logging.getLogger(__name__)


def getTable(
    session: Any,
    url_data: str,
    index: int,
    indices: list,
    delimiter: str = "//tbody",
    param: Any = None,
    debug: bool = False,
) -> tuple:
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
    if page.status_code != 200:
        logger.error(f"cannot logging to {url_data}")
        sys.exit(1)
    page.raise_for_status()
    logger.debug(f"connect: {page.url}, {page.status_code}, {page.text}")
    logger.debug(f"index: {index}")
    logger.debug(f"indices: {indices}")
    logger.debug(f"params: {param}")
    # logger.debug(f"page.text={page.text} **")

    # Store the contents of the website under doc
    doc = lh.fromstring(page.content)

    # from php source
    # table : id datatable, tr id=row, thead, tbody, <td class="sorting_1"...>
    # Parse data that are stored between <tbody>..</tbody> of HTML
    tr_elements = doc.xpath(delimiter)  # '//tbody')

    # Create empty list ## Better to have a dict??
    jid = None
    Mid = None
    Mjid: dict[str, Any] = dict()
    Mdata: dict[str, Any] = dict()
    Found: dict[str, Any] = dict()

    if not tr_elements:
        return (Mdata, Mjid)

    logger.debug(f"detected tables[delimiter={delimiter}]: {tr_elements}")
    for i, t in enumerate(tr_elements[0]):
        logger.debug(f"{i}: content={t.text_content()}, type={type(t.text_content())}")
        for j, d in enumerate(t):
            logger.debug(f"\t{j}:{d.text_content()}")

    # For each row, store each first element (header) and an empty list
    for _i, t in enumerate(tr_elements[0]):
        name = t.text_content()
        # print(f"name[{i}]={name}")
        # get date ID status comment from sub element
        data = []
        for j, d in enumerate(t):
            jname = d.text_content()
            logger.debug(f"\t{j}:{name}")
            if j + 1 == index:
                if param:
                    jid = jname[jname.find("(") + 1 : jname.find(")")]
                    # print("jid=", jid)
                Mid = re.sub(" (.*)", "", jname)
            if j + 1 in indices:
                data.append(jname)
        # shall check wether key is already defined for sanity
        if Mid == "-":
            logger.warning(f"{name} index: no entry")
        elif Mid is not None:
            if Mid in Mdata:
                Found[Mid] = Found[Mid] + 1
                Mid = f"{Mid}_{Found[Mid]}"
            else:
                Found[Mid] = 0
            Mdata[Mid] = data
            # print(f"Mdat[{Mid}] = {data}")
            Mjid[Mid] = jid

    # Mids = sorted(set(Mids)) #uniq only: list(set(Mids))
    logger.debug(f"Data found: Mdata={Mdata}, jid={Mjid}")
    return (Mdata, Mjid)


def getMaterial(
    session: Any,
    materialID: int | None,
    url_materials: str,
    Mats: dict,
    debug: bool = False,
) -> None:
    """get material"""
    # logger.info(f'getMaterial({materialID})')

    if materialID is None:
        r = session.post(url_materials, data={"compact:": "on", "formsubmit": "OK"}, verify=True)
        r.raise_for_status()
        # print("post Material: ", r.url, r.status_code)
        html = lh.fromstring(r.text.encode(r.encoding))
        sigmas = html.xpath('//input[@name="CONDUCTIVITE"]/@value')
        elasticlimits = html.xpath('//input[@name="LE"]/@value')
        refs = html.xpath('//input[@name="REF"]/@value')
        nuances = html.xpath('//input[@name="NUANCE"]/@value')
        if len(Mats.keys()) != len(refs) - 1:
            logger.debug(f"Materials in main list: {len(refs) - 1}")

        for i, ref in enumerate(refs):
            # ref is lxml.etree._ElementUnicodeResult
            if ref not in Mats and "finir" not in ref:
                logger.debug(
                    f"ref: {ref}, type: {type(ref)}, sigma: {sigmas[i]}, elasticlimit: {elasticlimits[i]}"  # noqa: E501
                )
                Mats[ref] = GObject(
                    str(ref),
                    "",
                    "",
                    {
                        "sigma0": str(sigmas[i]),
                        "rpe": str(elasticlimits[i]),
                        "nuance": str(nuances[i]),
                    },
                    "helix",
                    "Unknown",
                )
    else:
        r = session.post(
            url_materials,
            data={"REF": materialID, "compact:": "on", "formsubmit": "OK"},
            verify=True,
        )
        r.raise_for_status()
        html = lh.fromstring(r.text.encode(r.encoding))
        conductivity = html.xpath('//input[@name="CONDUCTIVITE"]/@value')[-1]
        elasticlimit = html.xpath('//input[@name="LE"]/@value')[-1]
        nuance = html.xpath('//input[@name="NUANCE"]/@value')[-1]
        # print(materialID, nuance)
        if materialID not in Mats:
            Mats[materialID] = GObject(
                str(materialID),
                "",
                "",
                {
                    "sigma0": str(conductivity),
                    "rpe": str(elasticlimit),
                    "nuance": nuance,
                },
                "helix",
                "Unknown",
            )


def getPartCADref(
    session: Any,
    url_data: str,
    Parts: dict,
    params: dict | None = None,
    part_type: str = "helix",
    debug: bool = False,
) -> bool:
    """get cadref and material for parts

    Args:
        session: requests session
        url_data: URL to fetch data from
        Parts: dictionary to store part data
        params: optional parameters dict for the POST request
        part_type: type of part ("helix" or "ring"), defaults to "helix"
        debug: enable debug logging

    Returns:
        True if successful
    """

    if params is None:
        params = {"REF": "", "compact": "on", "formsubmit": "OK"}

    res = session.post(url=url_data, data=params, verify=True)
    if res.status_code != 200:
        logger.error(f"getPartCADref: cannot logging to {url_data}")
        res.raise_for_status()
    logger.debug(f"getPartCADref: status={res.status_code}, res={res.text}")

    doc = lh.fromstring(res.content)

    delimiter = "//tbody"
    tr_elements = doc.xpath(delimiter)
    logger.debug(f"getPartCADref: detected tables[delimiter={delimiter}]: {tr_elements}")
    for _i, t in enumerate(tr_elements[0]):
        name = []
        for j, d in enumerate(t):
            name.append(d.text_content())
            logger.debug(f"\t{j}:{name[-1]}")
        # Store part data with part_type metadata
        Parts[name[0]] = [name[1], name[-1].split()[0], name[2], part_type]  # Ebauche

    logger.debug("getPartCADref:")
    for key in Parts:
        logger.debug(f"{key}: {Parts[key]}")
    return True


def getRingCADref(session: Any, url_data: str, Parts: dict, debug: bool = False) -> bool:
    """get CAD ref and material for rings from Bague.php

    Rings table columns: ID, Référence, Référence BE, Matériaux, Fichiers
    Data is in //*[@id="datatable"]/tbody
    We store: Référence (Name), Référence BE (CAD Ref), Matériaux (Material)

    Args:
        session: requests session
        url_data: URL to fetch data from (Bague.php)
        Parts: dictionary to store part data
        debug: enable debug logging

    Returns:
        True if successful
    """

    # First, check if there's a JSON endpoint via AJAX
    # DataTables often uses server-side processing with a separate endpoint
    res = session.get(url=url_data, verify=True)
    if res.status_code != 200:
        logger.error(f"getRingCADref: cannot connect to {url_data}")
        res.raise_for_status()
    logger.debug(f"getRingCADref: GET status={res.status_code}")

    doc = lh.fromstring(res.content)

    # Look for table with id="datatable" and check for AJAX data source
    scripts = doc.xpath("//script/text()")
    ajax_url = None

    for script in scripts:
        if "datatable" in script.lower():
            logger.debug(
                f"getRingCADref: DataTables script content (first 2000 chars): {script[:2000]}"  # noqa: E501
            )

            if "ajax" in script.lower() or "data" in script.lower():
                # Look for ajax URL pattern
                import re

                # Common patterns in DataTables configuration
                patterns = [
                    r'sAjaxSource\s*:\s*["\']([^"\']+)["\']',  # sAjaxSource: '?get' or sAjaxSource: 'file.php'  # noqa: E501
                    r'ajax\s*:\s*{\s*["\']?url["\']?\s*:\s*["\']([^"\']+)["\']',  # ajax: { url: "file.php" }  # noqa: E501
                    r'ajax\s*:\s*["\']([^"\']+\.php[^"\']*)["\']',  # ajax: "file.php"
                    r'data\s*:\s*["\']([^"\']+\.php[^"\']*)["\']',  # data: "file.php"
                    r'["\']url["\']\s*:\s*["\']([^"\']+\.php[^"\']*)["\']',  # "url": "file.php"  # noqa: E501
                    r'src\s*:\s*["\']([^"\']+\.php[^"\']*)["\']',  # src: "file.php"
                ]

                for pattern in patterns:
                    match = re.search(pattern, script, re.IGNORECASE)
                    if match:
                        ajax_url = match.group(1)
                        logger.debug(
                            f"getRingCADref: Found AJAX URL with pattern '{pattern[:50]}...': {ajax_url}"  # noqa: E501
                        )
                        break
                if ajax_url:
                    break

    # If no AJAX URL found, try common DataTables endpoints
    if not ajax_url:
        logger.debug("getRingCADref: No AJAX URL found in scripts, trying common endpoints")
        # Common DataTables data endpoints
        common_endpoints = ["data.php", "getData.php", "getBague.php", "bague_data.php"]

        for endpoint in common_endpoints:
            from urllib.parse import urljoin

            test_url = urljoin(url_data, endpoint)
            logger.debug(f"getRingCADref: Trying common endpoint: {test_url}")

            try:
                test_res = session.get(test_url, verify=True)
                if test_res.status_code == 200:
                    # Check if it looks like JSON
                    try:
                        import json

                        test_data = json.loads(test_res.text)
                        if "data" in test_data:
                            ajax_url = endpoint
                            logger.debug(f"getRingCADref: Found working endpoint: {endpoint}")
                            break
                    except json.JSONDecodeError:
                        continue
            except (OSError, ValueError, RuntimeError) as e:
                logger.debug(f"getRingCADref: Endpoint {endpoint} failed: {e}")
                continue

    if ajax_url:
        # Construct full URL if relative
        from urllib.parse import urljoin

        full_ajax_url = urljoin(url_data, ajax_url)
        logger.debug(f"getRingCADref: Fetching from AJAX endpoint: {full_ajax_url}")

        # Fetch data from AJAX endpoint
        ajax_res = session.get(full_ajax_url, verify=True)
        if ajax_res.status_code == 200:
            # Try JSON first
            try:
                import json

                data = json.loads(ajax_res.text)

                # DataTables can use 'data' or 'aaData' key
                data_key = "data" if "data" in data else "aaData" if "aaData" in data else None

                if data_key and isinstance(data[data_key], list):
                    logger.debug(
                        f"getRingCADref: Processing {len(data[data_key])} rows from JSON (key: {data_key})"  # noqa: E501
                    )

                    for idx, row in enumerate(data[data_key]):
                        logger.debug(f"getRingCADref: row {idx}: {row}")

                        # Row can be dict or list - handle both formats
                        if isinstance(row, dict):
                            # Use named keys (REF, REFBE) or string numeric keys ('2', '3', '4', '5')  # noqa: E501
                            name = str(row.get("REF", row.get("2", ""))).strip()  # Référence
                            cad_ref = str(
                                row.get("REFBE", row.get("3", ""))
                            ).strip()  # Référence BE
                            material_raw = str(row.get("4", "")).strip()  # Matériaux (HTML)
                        elif isinstance(row, list) and len(row) >= 6:
                            # Array format: [ID, Date, Référence, Référence BE, Matériaux, Fichiers]  # noqa: E501
                            name = str(row[2]).strip()  # Référence
                            cad_ref = str(row[3]).strip()  # Référence BE
                            material_raw = str(row[4]).strip()  # Matériaux
                        else:
                            logger.warning(
                                f"getRingCADref: Unexpected row format at index {idx}: {type(row)}"  # noqa: E501
                            )
                            continue

                        # Extract material text from HTML anchor if present
                        if material_raw and "<a" in material_raw:
                            import re

                            match = re.search(r">([^<]+)</a>", material_raw)
                            material = match.group(1) if match else material_raw
                        else:
                            material = material_raw

                        # Geometry is derived from CAD Ref by removing suffix letter
                        # Remove 2 chars if ends with "-X" (e.g., "HL-27-031-C" → "HL-27-031")  # noqa: E501
                        # Remove 1 char if ends with "X" only (e.g., "HL-27-034C" → "HL-27-034")  # noqa: E501
                        if re.search(r"-[A-Za-z]$", cad_ref):
                            geometry = cad_ref[:-2]  # Remove "-X"
                        elif re.search(r"[A-Za-z]$", cad_ref):
                            geometry = cad_ref[:-1]  # Remove "X"
                        else:
                            geometry = cad_ref  # No suffix letter

                        if name:
                            Parts[name] = [cad_ref, material, geometry, "ring"]
                            logger.debug(
                                f"getRingCADref: stored ring '{name}': {Parts[name]}"  # noqa: E501
                            )

                    ring_count = sum(1 for v in Parts.values() if len(v) > 3 and v[3] == "ring")
                    logger.debug(
                        f"getRingCADref: total rings stored from AJAX JSON: {ring_count}"  # noqa: E501
                    )
                    return True
                else:
                    logger.debug(f"getRingCADref: JSON response keys: {list(data.keys())}")
                    raise json.JSONDecodeError("No 'data' or 'aaData' key found", "", 0)

            except (json.JSONDecodeError, KeyError):
                # Not JSON, try parsing as HTML
                logger.debug("getRingCADref: AJAX response is not JSON, trying HTML parsing")

                ajax_doc = lh.fromstring(ajax_res.content)
                tbody = ajax_doc.xpath('//*[@id="datatable"]/tbody')

                if tbody:
                    rows = tbody[0].xpath(".//tr")
                    logger.debug(f"getRingCADref: found {len(rows)} rows in AJAX HTML response")

                    for i, row in enumerate(rows):
                        cells = row.xpath(".//td")
                        if not cells or len(cells) < 5:
                            logger.debug(
                                f"getRingCADref: row {i} has {len(cells)} cells, need >= 5"  # noqa: E501
                            )
                            continue

                        # Extract cell data: ID, Date, Référence, Référence BE, Matériaux, Fichiers  # noqa: E501
                        # Column 4 (Matériaux) may have <a> tag, so get text_content
                        cell_data = [cell.text_content().strip() for cell in cells]

                        logger.debug(f"getRingCADref: row {i} data: {cell_data}")

                        # cell_data[0]=ID, [1]=Date, [2]=Référence, [3]=Référence BE, [4]=Matériaux, [5]=Fichiers  # noqa: E501
                        if len(cell_data) >= 5 and cell_data[2]:
                            name = cell_data[2]  # Référence
                            cad_ref = cell_data[3]  # Référence BE
                            material = cell_data[4]  # Matériaux (text from <a> tag)
                            geometry = cell_data[5] if len(cell_data) > 5 else ""  # Fichiers

                            Parts[name] = [cad_ref, material, geometry, "ring"]
                            logger.debug(
                                f"getRingCADref: stored ring '{name}': {Parts[name]}"  # noqa: E501
                            )

                    ring_count = sum(1 for v in Parts.values() if len(v) > 3 and v[3] == "ring")
                    logger.debug(
                        f"getRingCADref: total rings stored from AJAX HTML: {ring_count}"  # noqa: E501
                    )
                    return True

    # Fallback: look for tbody in the initial response using the specific XPath
    logger.debug("getRingCADref: Looking for data in //*[@id='datatable']/tbody")
    tbody = doc.xpath('//*[@id="datatable"]/tbody')

    if tbody:
        rows = tbody[0].xpath(".//tr")
        logger.debug(f"getRingCADref: found tbody with {len(rows)} rows")

        for i, row in enumerate(rows):
            cells = row.xpath(".//td")
            if not cells or len(cells) < 4:
                logger.debug(f"getRingCADref: row {i} has {len(cells)} cells, need >= 4")
                continue

            # Extract cell text: ID, Référence, Référence BE, Matériaux, Fichiers
            cell_data = [cell.text_content().strip() for cell in cells]

            logger.debug(f"getRingCADref: row {i} data: {cell_data}")

            # cell_data[0] = ID, [1] = Référence, [2] = Référence BE, [3] = Matériaux, [4] = Fichiers  # noqa: E501
            if len(cell_data) >= 4 and cell_data[1]:
                name = cell_data[1]  # Référence
                cad_ref = cell_data[2]  # Référence BE
                material = cell_data[3]  # Matériaux
                geometry = cell_data[4] if len(cell_data) > 4 else ""  # Fichiers

                Parts[name] = [cad_ref, material, geometry, "ring"]
                logger.debug(f"getRingCADref: stored ring '{name}': {Parts[name]}")

        ring_count = sum(1 for v in Parts.values() if len(v) > 3 and v[3] == "ring")
        logger.debug(f"getRingCADref: total rings stored from tbody: {ring_count}")
        return True

    logger.warning("getRingCADref: Could not find ring data in //*[@id='datatable']/tbody")
    return False


def getMagnetPart(
    session: Any,
    magnet: str,
    url_helices: str,
    Magnets: dict,
    url_materials: str,
    Parts: dict,
    Mats: dict,
    url_confs: str,
    Confs: dict = None,
    datadir: str = ".",
    save: bool = False,
    debug: bool = False,
) -> None:
    """get parts for a given magnet"""
    logger.info(f"getMagnetPart({magnet})")
    params_helix = (("ref", magnet),)

    hindices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    logger.debug(f"getTable for : magnet={magnet}, hindices={hindices}, params={params_helix}")
    res = getTable(session, url_helices, 1, hindices, param=params_helix, debug=False)
    logger.debug(f"getMagnetPart: res={res}")
    helices: Any = ()
    jid = None
    if res:
        helices = res[0]
        jid = res[1]
    logger.debug(f"getMagnetPart: helices={helices}")

    if magnet not in Parts:
        Parts[magnet] = []
    logger.debug(f"{magnet}: jid={jid}")
    for data in helices:
        # print(f"data: {data}")
        # print(f"helices: {len(helices[data])}")
        for i in range(len(helices[data])):
            ids = helices[data][i].split(" / ")
            # print(f'ids[{i}={ids}')
            if ids[0] != "-":
                [partID, materialID] = ids
                Parts[magnet].append([i, partID])
                Magnets[magnet].addPart(partID)
                getMaterial(session, materialID, url_materials, Mats, debug)
                logger.debug(f"{partID}: {materialID}")

    # get MagConfFile
    hindices = [19]
    logger.debug(
        f"getTable for conf files: magnet={magnet}, hindices={hindices}, params={params_helix}"  # noqa: E501
    )
    res = getTable(session, url_helices, 1, hindices, param=params_helix, debug=False)
    logger.debug(f"res={res}")
    data = res[0][magnet]
    id = res[1][magnet]
    data = data[0].replace("\t", "")
    files = data.split()
    if files == ["Pas", "de", "fichiers"]:
        files = []

    Confs[magnet] = files
    if save:
        for file in files:
            logger.info(f"file={file}")
            r = download(
                session,
                url_data=url_confs,
                param={"ID": id, "NAME": file},
                debug=debug,
            )
            logger.debug(f"conf[{magnet}]: {file}, r={r}")

            filename = f"{magnet}_{file}"
            if datadir != ".":
                filename = f"{datadir}/{filename}"
            with open(filename, "w") as f:
                f.write(r)

    pass


def getSiteRecord(
    session: Any,
    url_data: str,
    ID: str,
    Sites: dict,
    url_downloads: str,
    debug: bool = False,
) -> None:
    """get records for a given ID"""
    # looger.info(f'getSiteRecord({ID})')

    # To get files for ID
    params_links = (("ref", re.sub("_\\d+", "", ID)), ("link", ""))

    r = session.get(url=url_data, params=params_links, verify=True)
    logger.debug(
        f"data: url={r.url}, status={r.status_code}, encoding={r.encoding}, text={r.text}"  # noqa: E501
    )
    if r.status_code != 200:
        logger.error(f"error {r.status_code} loading {url_data}")
        sys.exit(1)
    r.raise_for_status()

    # Get list of records attached to ID
    for f in r.text.split("<br>"):
        if f and "~" not in f:
            replace_str = "<a href=" + "'" + url_downloads + "?file="

            data = f.replace(replace_str, "").replace("</a>", "").replace("'>", ": ").split(": ")
            link = data[0].replace(" ", "%20")
            link = re.sub("<a?(.*?)file=", "", link, flags=re.DOTALL)
            site = link.replace("../../../", "")
            site = re.sub("/.*txt", "", site)

            housing = link.replace("../../../", "").split("/")[0]
            # print(f"link={link}, site={site}, housing={housing}")
            logger.debug(f"data={data}, link={link}, site={site}, housing={housing}")

            tformat = "%Y.%m.%d - %H:%M:%S"
            timestamp = datetime.datetime.strptime(data[1].replace(".txt", ""), tformat)
            # print(f'timestamp: {timestamp}, {type(timestamp)}')

            # # Download a specific file
            # params_downloads = 'file=%s&download=1' % link
            # html = download(session, url_downloads, param=params_downloads, link=link)

            # lines = html.split('\n')[0] # get 1st line
            # lines_items = lines.split('\t')

            # actual_id = None
            # if len(lines_items) >= 2:
            #     actual_id = lines_items[1]
            # logger.debug(f'{magnetID}: actual_id={actual_id}, site={site} link={link}, param={params_downloads}')  # noqa: E501

            # if not actual_id:
            #     logger.debug("%s: no name defined for Magnet" % link)

            # else:
            record = MRecord(timestamp, housing, ID, link)
            created_at = Sites[ID]["commissioned_at"]
            stopped_at = Sites[ID]["decommissioned_at"]
            if record.timestamp < created_at or record.timestamp > stopped_at:
                logger.debug(f"{ID}: {record} dropped")
            else:
                Sites[ID]["records"].append(record)
            logger.debug(f"{ID}: site={site} link={link}")
            # data = record.getData(session, url_downloads, save)

            # if actual_id != magnetID:
            #     print(f"record: incoherent data magnetID {magnetID} actual_id: {actual_id} - {timestamp}, {site} {link}" )  # noqa: E501
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
            logger.debug(f"getSiteRecords({ID}): f={f}")


def getCirrusFiles(
    session: Any,
    url_cirrus: str,
    feed: str = "A1",
    datadir: str = ".",
    save: bool = False,
    debug: bool = False,
) -> dict:
    """
    Get logs and XML configuration files from cirrus.php page

    The cirrus.php page displays a table with links like "Logs A1" and "XML A1"
    that point to directories containing the actual log and XML files.

    Args:
        session: requests session
        url_cirrus: URL to cirrus.php
        feed: cirrus feed name (A1, A2, A3, A4, etc.)
        datadir: directory to save files
        save: whether to save files to disk
        debug: enable debug logging

    Returns:
        dict with 'logs' and 'xmls' lists containing file information
    """
    import os
    from urllib.parse import urljoin

    logger.info(f"Loading cirrus files for feed: {feed}")

    # Request the cirrus page
    try:
        r = session.get(url=url_cirrus, verify=True)
        r.raise_for_status()
        logger.debug(f"cirrus: url={r.url}, status={r.status_code}")
    except (OSError, RuntimeError) as e:
        logger.error(f"Error loading cirrus page: {e}")
        return {"logs": [], "xmls": []}

    if r.status_code != 200:
        logger.error(f"error {r.status_code} loading {url_cirrus}")
        return {"logs": [], "xmls": []}

    # Parse the HTML response
    doc = lh.fromstring(r.content)

    cirrus_files: dict[str, list] = {"logs": [], "xmls": []}

    # Look for links in the card-body div structure
    # Structure: div.card-body > div.col-lg-12 > div.list-group > a
    card_body_links = doc.xpath(
        '//div[@class="card-body"]//div[@class="col-lg-12"]//div[@class="list-group"]//a[@href]'
    )

    logs_url = None
    xml_url = None

    for link_element in card_body_links:
        href = link_element.get("href")
        text = link_element.text_content().strip()

        if not href:
            continue

        logger.debug(f"Found table link: text='{text}', href={href}")

        # Look for "Logs A1", "Logs A2", etc.
        if text.lower() == f"logs {feed.lower()}":
            logs_url = urljoin(url_cirrus, href)
            logger.info(f"Found logs link for {feed}: {logs_url}")

        # Look for "XML A1", "XML A2", etc.
        elif text.lower() == f"xml {feed.lower()}":
            xml_url = urljoin(url_cirrus, href)
            logger.info(f"Found XML link for {feed}: {xml_url}")

    # Fetch log files if logs URL was found
    if logs_url:
        try:
            logger.debug(f"Fetching logs from: {logs_url}")
            logs_response = session.get(logs_url, verify=True)
            logs_response.raise_for_status()

            logs_doc = lh.fromstring(logs_response.content)
            log_links = logs_doc.xpath("//a[@href]")

            for log_link in log_links:
                log_href = log_link.get("href")
                log_text = log_link.text_content().strip()

                if not log_href:
                    continue

                # Skip "Voir" links but keep "Voir Propre"
                if log_text == "Voir":
                    logger.debug(f"Skipping 'Voir' link: {log_href}")
                    continue

                # Check if it's a log file
                if log_href.endswith(".log") or ".log" in log_href:
                    log_info = {
                        "name": log_text or log_href.split("/")[-1],
                        "url": urljoin(logs_url, log_href),
                        "feed": feed,
                    }
                    cirrus_files["logs"].append(log_info)
                    logger.debug(f"Found log file: {log_info['name']}")

                    if save:
                        # Download the log file
                        try:
                            log_file_response = session.get(log_info["url"], verify=True)
                            log_file_response.raise_for_status()

                            filename = f"{feed}_{log_info['name']}"
                            if datadir != ".":
                                filename = os.path.join(datadir, filename)

                            with open(filename, "w") as f:
                                f.write(log_file_response.text)
                            logger.info(f"Saved log file: {filename}")
                        except (OSError, RuntimeError) as e:
                            logger.error(f"Error downloading log {log_info['url']}: {e}")

        except (OSError, RuntimeError) as e:
            logger.error(f"Error fetching logs from {logs_url}: {e}")
    else:
        logger.warning(f"No logs link found for feed {feed}")

    # Fetch XML files if XML URL was found
    if xml_url:
        try:
            logger.debug(f"Fetching XMLs from: {xml_url}")
            xml_response = session.get(xml_url, verify=True)
            xml_response.raise_for_status()

            xml_doc = lh.fromstring(xml_response.content)
            xml_links = xml_doc.xpath("//a[@href]")

            for xml_link in xml_links:
                xml_href = xml_link.get("href")
                xml_text = xml_link.text_content().strip()

                if not xml_href:
                    continue

                # Skip "Voir" links but keep "Voir Propre"
                if xml_text == "Voir":
                    logger.debug(f"Skipping 'Voir' link: {xml_href}")
                    continue

                # Check if it's an XML file
                if xml_href.endswith(".xml") or ".xml" in xml_href:
                    xml_info = {
                        "name": xml_text or xml_href.split("/")[-1],
                        "url": urljoin(xml_url, xml_href),
                        "feed": feed,
                    }
                    cirrus_files["xmls"].append(xml_info)
                    logger.debug(f"Found XML file: {xml_info['name']}")

                    if save:
                        # Download the XML file
                        try:
                            xml_file_response = session.get(xml_info["url"], verify=True)
                            xml_file_response.raise_for_status()

                            filename = f"{feed}_{xml_info['name']}"
                            if datadir != ".":
                                filename = os.path.join(datadir, filename)

                            with open(filename, "w") as f:
                                f.write(xml_file_response.text)
                            logger.info(f"Saved XML file: {filename}")
                        except (OSError, RuntimeError) as e:
                            logger.error(f"Error downloading XML {xml_info['url']}: {e}")

        except (OSError, RuntimeError) as e:
            logger.error(f"Error fetching XMLs from {xml_url}: {e}")
    else:
        logger.warning(f"No XML link found for feed {feed}")

    logger.info(
        f"Found {len(cirrus_files['logs'])} log files and {len(cirrus_files['xmls'])} XML files for feed {feed}"  # noqa: E501
    )

    return cirrus_files
