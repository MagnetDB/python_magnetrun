#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""HMagnet Object"""

import json
import deserialize

class HMagnet:
    """
    name
    cadref
    GObjects: list of Objects IDs (aka rings, helices, current leads)
    records: timestamps of txt files
    rapidrecords: timestamps of tdms files
    MAGfile(s)
    status: Dead/Alive
    index
    """

    def __init__(self, name: str, cadref: str, GObjects: list, records: list, rapidrecords: list, MAGfile: list, status: str, index: int):
        """defaut constructor"""
        self.name = name
        self.cadref = cadref
        self.GObjects = GObjects
        self.records = records
        self.rapidrecords = rapidrecords
        self.MAGfile = MAGfile
        self.status = status
        self.index = index

    def __repr__(self):
        """
        representation of object
        """
        return "%s(name=%r, cadref=%r, GObjects=%r, records=%r, rapidrecords=%r, MAGfile=%r, status=%r, index=%r)" % \
            (self.__class__.__name__,
             self.name,
             self.cadref,
             self.GObjects,
             self.records,
             self.rapidrecords,
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
        self.records = records

    def addRecord(self, record):
        """add a record to records list (aka txt files)"""
        self.records.append(record)

    def getRecords(self):
        """get records list (aka txt files)"""
        return self.records

    def setRapidRecords(self, records):
        """set records list (aka tdms files)"""
        self.rapidrecords = records

    def addRapidRecord(self, record):
        """add a rpid record to records list (aka tdms files)"""
        self.rapidrecords.append(record)

    def getRapidRecords(self):
        """get rapid records list (aka tdms files)"""
        return self.rapidrecords

    def setGObjects(self, GObjects):
        """set list of GObjects composing Magnets"""
        self.GObjects = GObjects

    def addGObject(self, GObject):
        """add GObject to list of GObjects composing Magnets"""
        self.GObjects.append(GObject)

    def getGObjects(self):
        """get list of GObjects composing Magnets"""
        return self.GObjects

    def getGObjects(self, category):
        """
        get list of GObjects composing Magnets
        from a given category
        """
        selected = [gobject for gobject in self.GObjects if gobject.getCategory() == category]
        return selected

    def getGObjects(self):
        """get list of GObjects composing Magnets"""
        return self.GObjects

    def setMAGfile(self, MAGfile):
        """set MAGfile configuration file(s)"""
        self.MAGfile = MAGfile

    def getMAGfile(self):
        """get MAGfile configuration file(s)"""
        return self.MAGfile

    # def download(self, session, url):
    # """Download MAGfile"""

    def to_json(self):
        """
        convert to json
        """
        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
