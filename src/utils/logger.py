#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities
"""

import logging
import os


def setlogger(path):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    
    # Console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
