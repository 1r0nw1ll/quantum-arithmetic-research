import urllib.request
import json


def fetch(url):
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())
    except Exception:
        pass
