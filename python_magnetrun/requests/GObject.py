#!/usr/bin/env python3

"""Magnet component Object"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GObject:
    """
    name
    cadref
    geometry: yaml file
    material: dict of physical properties (sigma0, rpe) only
    cf json mat in feelp to complete ??
    category: Helix, Ring, Current Lead, Bitter
    status: Dead/Alive
    """

    def __init__(
        self,
        name: str,
        cadref: str,
        geofile: str,
        material: dict,
        category: str,
        status: str,
    ) -> None:
        """default constructor"""
        self.name = name
        self.cadref = cadref
        self.geofile = geofile
        self.material = material
        self.category = category
        self.status = status

    def __repr__(self) -> str:
        """
        representation of object
        """
        return f"{self.__class__.__name__}(name={self.name!r}, cadref={self.cadref!r}, geofile={self.geofile!r}, material={self.material!r}, category={self.category!r}, status={self.status!r})"

    def setCadref(self, cadref: str) -> None:
        """set cadref"""
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

    def setCategory(self, category: str) -> None:
        """set category"""
        self.category = category

    def getCategory(self) -> str:
        """get category"""
        return self.category

    def getMaterial(self) -> dict:
        """get Material"""
        return self.material

    def getMaterialProperty(self, mproperty: str) -> dict:
        """get Material Property"""
        return self.material[mproperty]

    def setMaterial(self, material: dict) -> None:
        """set Material"""
        self.material = material

    def setMaterialProperty(self, mproperty: str, mval: Any) -> None:
        """set Material Property"""
        self.material[mproperty] = mval

    def to_json(self) -> str:
        """
        convert to json
        """
        from . import deserialize

        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
