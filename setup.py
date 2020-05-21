#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.rst", "r") as ff:
    long_description = ff.read()

setuptools.setup(
    name = "bsfc",
    version = "1.0.1",
    author = "Francesco Sciortino and Norman Cao",
    author_email = "sciortino@psfc.mit.edu",
    description = "Bayesian Spectral Fitting Code (BSFC)",
    long_description=long_description,
    #long_description_content_type="text/markdown",
    maintainer = "Francesco Sciortino",
    maintainer_email = "sciortino@psfc.mit.edu",
    url = "https://github.com/Maplenormandy/bsfc",
    packages = setuptools.find_packages(),
    requires = ["matplotlib","scipy", "numpy (>=1.5)","pymultinest","emcee","MDSplus",
                "gptools","eqtools","profiletools"],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
