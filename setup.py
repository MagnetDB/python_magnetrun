#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "itertools",
    "iapws",
    "itertools",
    "lxml",
    "matplotlib",
    "seaborn",
    "numpy",
    "scipy",
    "pandas",
    "nptdms",
    "pint",
    "requests",
    "statsmodel",
    "tabulate",
    "natsort",
]

setup_requirements = []

test_requirements = [
    "pytest",
]

extras_requires = {
    "system": [
        "ruptures",
    ],
    "signal": ["ht", "clawpack"],
}

setup(
    author="Christophe Trophime",
    author_email="christophe.trophime@lncmi.cnrs.fr",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python MagnetRun contains utils to view and analyse Magnet runs",
    entry_points={
        "console_scripts": [
            "python_magnetrun=python_magnetrun.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="python_magnetrun",
    name="python_magnetrun",
    packages=find_packages(include=["python_magnetrun", "python_magnetrun.*"]),
    setup_requires=setup_requirements,
    extras_require=extras_requires,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Trophime/python_magnetrun",
    version="0.1.0",
    zip_safe=False,
)
