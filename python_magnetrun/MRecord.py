#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""Magnet Record Object"""

import json

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
    #     # need test-request.py
    #
    #     payload = {
    #         'email': email_address,
    #         'password': password
    #     }

    #     session = createSession(url_logging, payload)
    #
    #     params = 'file=%s&download=1' % self.link
    #     download(session, url_downloads, params, self.link, save=True, debug=False)


    def to_json(self):
        """
        convert to json
        """
        import deserialize
        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
