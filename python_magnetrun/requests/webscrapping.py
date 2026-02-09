#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import sys
import re
import datetime
import lxml.html as lh
import logging

logger = logging.getLogger(__name__)

# import jsonpickle
from .. import MRecord
from .. import GObject

from .connect import download

# for M1:
# table fileTreeDemo_1, ul, <li  class="file ext_txt">, <a href=.., rel="filename" /a> </li>


def getTable(
    session, url_data, index, indices, delimiter="//tbody", param=None, debug=False
):
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
    if debug:
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
    Mjid = dict()
    Mdata = dict()
    Found = dict()

    if not tr_elements:
        return (Mdata, Mjid)

    if debug:
        logger.debug(f"detected tables[delimiter={delimiter}]: {tr_elements}")
        for i, t in enumerate(tr_elements[0]):
            logger.debug(
                f"{i}: content={t.text_content()}, type={type(t.text_content())}"
            )
            for j, d in enumerate(t):
                logger.debug(f"\t{j}:{d.text_content()}")

    # For each row, store each first element (header) and an empty list
    for i, t in enumerate(tr_elements[0]):
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
        else:
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
    session, materialID: int | None, url_materials, Mats: dict, debug=False
):
    """get material"""
    # logger.info(f'getMaterial({materialID})')

    if materialID is None:
        r = session.post(
            url_materials, data={"compact:": "on", "formsubmit": "OK"}, verify=True
        )
        r.raise_for_status()
        # print("post Material: ", r.url, r.status_code)
        html = lh.fromstring(r.text.encode(r.encoding))
        sigmas = html.xpath('//input[@name="CONDUCTIVITE"]/@value')
        elasticlimits = html.xpath('//input[@name="LE"]/@value')
        refs = html.xpath('//input[@name="REF"]/@value')
        nuances = html.xpath('//input[@name="NUANCE"]/@value')
        if debug:
            if len(Mats.keys()) != len(refs) - 1:
                logger.debug("Materials in main list:", len(refs) - 1)

        for i, ref in enumerate(refs):
            # ref is lxml.etree._ElementUnicodeResult
            if ref not in Mats and "finir" not in ref:
                logger.debug(
                    f"ref: {ref}, type: {type(ref)}, sigma: {sigmas[i]}, elasticlimit: {elasticlimits[i]}"
                )
                Mats[ref] = GObject.GObject(
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
            Mats[materialID] = GObject.GObject(
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


def getPartCADref(session, url_data, Parts, debug: bool = False):
    """get cadref and material for parts"""

    params = {"REF": "", "compact": "on", "formsubmit": "OK"}
    res = session.post(url=url_data, data=params, verify=True)
    if res.status_code != 200:
        logger.error(f"getPartCADref: cannot logging to {url_data}")
        res.raise_for_status()
    logger.debug(f"getPartCADref: status={res.status_code}, res={res.text}")

    doc = lh.fromstring(res.content)

    delimiter = "//tbody"
    tr_elements = doc.xpath(delimiter)
    for i, t in enumerate(tr_elements[0]):
        name = []
        for j, d in enumerate(t):
            name.append(d.text_content())
            logger.debug(f"\t{j}:{name[-1]}")
        Parts[name[0]] = [name[1], name[-1].split()[0], name[2]]  # Ebauche

    logger.debug("getPartCADref:")
    for key in Parts:
        logger.debug(f"{key}: {Parts[key]}")
    return True


def getMagnetPart(
    session,
    magnet,
    url_helices,
    Magnets,
    url_materials,
    Parts,
    Mats,
    url_confs,
    Confs: dict = {},
    datadir: str = ".",
    save: bool = False,
    debug: bool = False,
):
    """get parts for a given magnet"""
    logger.info(f"getMagnetPart({magnet})")
    params_helix = (("ref", magnet),)

    hindices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    logger.debug(
        f"getTable for : magnet={magnet}, hindices={hindices}, params={params_helix}"
    )
    res = getTable(session, url_helices, 1, hindices, param=params_helix, debug=False)
    logger.debug(f"getMagnetPart: res={res}")
    helices = ()
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
        f"getTable for conf files: magnet={magnet}, hindices={hindices}, params={params_helix}"
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
            print(f"file={file}")
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


def getSiteRecord(session, url_data, ID, Sites, url_downloads, debug=False):
    """get records for a given ID"""
    # looger.info(f'getSiteRecord({ID})')

    # To get files for ID
    params_links = (("ref", re.sub("_\\d+", "", ID)), ("link", ""))

    r = session.get(url=url_data, params=params_links, verify=True)
    logger.debug(
        f"data: url={r.url}, status={r.status_code}, encoding={r.encoding}, text={r.text}"
    )
    if r.status_code != 200:
        logger.error(f"error {r.status_code} loading {url_data}")
        sys.exit(1)
    r.raise_for_status()

    # Get list of records attached to ID
    for f in r.text.split("<br>"):
        if f and "~" not in f:
            replace_str = "<a href=" + "'" + url_downloads + "?file="

            data = (
                f.replace(replace_str, "")
                .replace("</a>", "")
                .replace("'>", ": ")
                .split(": ")
            )
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
            # logger.debug(f'{magnetID}: actual_id={actual_id}, site={site} link={link}, param={params_downloads}')

            # if not actual_id:
            #     logger.debug("%s: no name defined for Magnet" % link)

            # else:
            record = MRecord.MRecord(timestamp, housing, ID, link)
            created_at = Sites[ID]["commissioned_at"]
            stopped_at = Sites[ID]["decommissioned_at"]
            if record.timestamp < created_at or record.timestamp > stopped_at:
                logger.debug(f"{ID}: {record} dropped")
            else:
                Sites[ID]["records"].append(record)
            logger.debug(f"{ID}: site={site} link={link}")
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
            logger.debug(f"getSiteRecords({ID}): f={f}")
