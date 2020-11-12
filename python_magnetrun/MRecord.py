#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""Magnet Record Object"""

import json
import deserialize

class MRecord:
    """
    timestamp
    site
    link
    """

    def __init__(self, timestamp, site: str, link: str):
        """default constructor"""
        self.timestamp = timestamp
        self.site = site
        self.link = link

    def __repr__(self):
        """
        representation of object
        """
        return "%s(timestamp=%r, site=%r, link=%r)" % \
            (self.__class__.__name__,
             self.timestamp,
             self.site,
             self.link
            )

    def getTimestamp(self):
        """get timestamp"""
        return self.timestamp

    def getSite(self):
        """get site"""
        return self.site

    def getLink(self):
        """get link"""
        return self.link

    def setTimestamp(self, timestamp):
        """set timestamp"""
        self.timestamp = timestamp

    def setSite(self, site):
        """set Site"""
        self.site = site

    def setLink(self, link):
        """set Link"""
        self.link = link

    # def download(self, session, url):
    #     """download record"""
    #     params_downloads = 'file=%s&download=1' % self.link
    #     d = session.get(url=url, params=params_downloads)

    #     filename = self.link.replace('../../../','')
    #     filename = filename.replace('/','_').replace('%20','-')
    #     fo = open(filename, "w", newline='\n')
    #     fo.write(d.text)
    #     fo.close()

