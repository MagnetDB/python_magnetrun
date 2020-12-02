#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""HMagnet Object"""

import json

class HMagnet:
    """
    name
    cadref
    MAGfile(s)
    status: Dead/Alive
    index
    """

    def __init__(self, name: str, cadref: str, MAGfile: list, status: str, index: int):
        """defaut constructor"""
        self.name = name
        self.cadref = cadref
        self.MAGfile = MAGfile
        self.status = status
        self.index = index

    def __repr__(self):
        """
        representation of object
        """
        return "%s(name=%r, cadref=%r, MAGfile=%r, status=%r, index=%r)" % \
            (self.__class__.__name__,
             self.name,
             self.cadref,
             self.MAGfile,
             self.status,
             self.index
            )

    def setIndex(self, index):
        """set Index"""
        self.index = index

    def getIndex(self):
        """get index"""
        return self.index

    def setCadref(self, cadref):
        """set Cadref"""
        self.cadref = cadref

    def getCadref(self):
        """get cadref"""
        return self.cadref

    def setStatus(self, status):
        """set status"""
        self.status = status

    def getStatus(self):
        """get status"""
        return self.status

    def setRecords(self, records):
        """set records list (aka txt files)"""

    def addRecord(self, record):
        """add a record to records list (aka txt files)"""

    def getRecords(self):
        """get records list (aka txt files)"""

    def setRapidRecords(self, records):
        """set records list (aka tdms files)"""

    def addRapidRecord(self, record):
        """add a rpid record to records list (aka tdms files)"""

    def getRapidRecords(self):
        """get rapid records list (aka tdms files)"""

    def setGObjects(self, GObjects):
        """set list of GObjects composing Magnets"""

    def addGObject(self, GObject):
        """add GObject to list of GObjects composing Magnets"""

    def getGObjects(self):
        """get list of GObjects composing Magnets"""

    def getGObjects(self, category):
        """
        get list of GObjects composing Magnets
        from a given category
        """

    def getGObjects(self):
        """get list of GObjects composing Magnets"""

    def setMAGfile(self, MAGfile):
        """set MAGfile configuration file(s)"""
        self.MAGfile = MAGfile

    def getMAGfile(self):
        """get MAGfile configuration file(s)"""
        return self.MAGfile

    # def download(self):
    #     """Download MAGfile"""
    #     # need test-request.py
    #
    #     payload = {
    #         'email': email_address,
    #         'password': password
    #     }

    #     session = createSession(url_logging, payload)
    #
    #     url = base_url + "/" + "downloadM.php"
    #     for f in self.MAGfile:
    #         params = ( ('ID', self.index), ('NAME', f), )
    #         download(session, , params, f, save=True, debug=False)
    # 

    def to_json(self):
        """
        convert to json
        """
        import deserialize
        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
