#!/usr/bin/env python3

"""HMagnet Object"""

import json
import logging

logger = logging.getLogger(__name__)


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

    def __repr__(self) -> str:
        """
        representation of object
        """
        return f"{self.__class__.__name__}(name={self.name!r}, cadref={self.cadref!r}, status={self.status!r}, parts={self.parts!r})"

    def setParts(self, parts: list) -> None:
        """set Parts"""
        if not self.parts:
            self.parts = parts

    def addPart(self, part: str) -> None:
        """add to Parts"""
        if part not in self.parts:
            self.parts.append(part)

    def getParts(self) -> list:
        """get parts"""
        return self.parts

    def setCadref(self, cadref: str) -> None:
        """set Cadref"""
        self.cadref = cadref

    def getCadref(self) -> str:
        """get cadref"""
        return self.cadref

    def setStatus(self, status: str) -> None:
        """set status"""
        self.status = status

    def getStatus(self) -> str:
        """get status"""
        return self.status

    def to_json(self) -> str:
        """
        convert to json
        """
        from . import deserialize

        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
