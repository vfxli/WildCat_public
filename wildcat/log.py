# -*- coding: utf-8 -*-
import datetime

_FILE = None

def init(path):
    global _FILE
    _FILE = path

def log(msg):
    line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if _FILE:
        with open(_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")