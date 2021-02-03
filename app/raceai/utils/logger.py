#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file logger.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-02-03 11:07

import logging
import sys
import os
from logging import handlers


_LEVELS_ = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

_LOG_LEVEL_ = 'info'
_LOG_FILENAME_ = '/tmp/raceai.log'


def race_set_loglevel(level):
    global _LOG_LEVEL_
    _LOG_LEVEL_ = level


def race_set_logfile(filename):
    global _LOG_FILENAME_
    _LOG_FILENAME_ = filename


class Logger(object):

    logger = None

    @staticmethod
    def init(filename, level='info', when='D', backCount=3, fmt='%(asctime)s %(levelname)-7s %(message)s'):
        Logger.logger = logging.getLogger(filename)
        Logger.logger.setLevel(_LEVELS_.get(level))

        console = logging.StreamHandler()
        logfile = handlers.TimedRotatingFileHandler(
                filename=filename,
                when=when,
                backupCount=backCount,
                encoding='utf-8')

        format = logging.Formatter(fmt)
        console.setFormatter(format)
        logfile.setFormatter(format)
        Logger.logger.addHandler(console)
        Logger.logger.addHandler(logfile)

    @staticmethod
    def prefix():
        if Logger.logger is None:
            Logger.init(_LOG_FILENAME_, _LOG_LEVEL_)
        frame = sys._getframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        return '{} {}:{}'.format(filename, funcname, lineno)

    @staticmethod
    def debug(message):
        prefix = Logger.prefix()
        Logger.logger.debug(f'{prefix} {message}')

    @staticmethod
    def info(message):
        prefix = Logger.prefix()
        Logger.logger.info(f'{prefix} {message}')

    @staticmethod
    def warning(message):
        prefix = Logger.prefix()
        Logger.logger.warning(f'{prefix} {message}')

    @staticmethod
    def error(message):
        prefix = Logger.prefix()
        Logger.logger.error(f'{prefix} {message}')

    @staticmethod
    def critical(message):
        prefix = Logger.prefix()
        Logger.logger.critical(f'{prefix} {message}')
