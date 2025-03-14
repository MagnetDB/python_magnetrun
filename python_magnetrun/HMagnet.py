#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""HMagnet Object"""

import json


class HMagnet:
    """
    name
    cadref
    status: Dead/Alive
    parts
    """

    def __init__(self, name: str, cadref: str, status: str, parts: list) -> None:
        """defaut constructor"""
        self.name = name
        self.cadref = cadref
        self.status = status
        self.parts = parts

    def __repr__(self):
        """
        representation of object
        """
        return "%s(name=%r, cadref=%r, status=%r, parts=%r)" % (
            self.__class__.__name__,
            self.name,
            self.cadref,
            self.status,
            self.parts,
        )

    def setParts(self, parts: list) -> None:
        """set Parts"""
        if not self.parts:
            self.parts = parts

    def addPart(self, part: str) -> None:
        """add to Parts"""
        if not part in self.parts:
            self.parts.append(part)

    def getParts(self):
        """get parts"""
        return self.parts

    def setCadref(self, cadref) -> None:
        """set Cadref"""
        self.cadref = cadref

    def getCadref(self) -> str:
        """get cadref"""
        return self.cadref

    def setStatus(self, status) -> None:
        """set status"""
        self.status = status

    def getStatus(self) -> str:
        """get status"""
        return self.status

    def to_json(self):
        """
        convert to json
        """
        from . import deserialize

        return json.dumps(
            self, default=deserialize.serialize_instance, sort_keys=True, indent=4
        )
