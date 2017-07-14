import cherrypy
from hlparser import HighlightedParser
import requests
import json
import argparse
import hashlib
import os
import shutil

class HighlightedServer(object):

    @cherrypy.expose
    def index(self):
        return "Highlighted server is running."

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def highlight(self):
        data = cherrypy.request.json

        highlighted_text = data['highlighted_text']
        new_filing_url = data['new_filing_url']
        ignore_money = bool(data['ignore_money']) if 'ignore_money' in data.keys() else True
        ignore_date = bool(data['ignore_date']) if 'ignore_date' in data.keys() else True
        fuzzy = bool(data['fuzzy']) if 'fuzzy' in data.keys() else False
        lsi = bool(data['LSI']) if 'LSI' in data.keys() else False
        min_ratio = int(data['min_ratio']) if 'min_ratio' in data.keys() else 75
        depth = int(data['depth']) if 'depth' in data.keys() else 0
        min_sentence_length = int(data['min_sentence_length']) if 'min_sentence_length' in data.keys() else 3
        num_search = int(data['num_search']) if 'min_sentence_length' in data.keys() else 3
        force = bool(data['force']) if 'force' in data.keys() else False
        name = hashlib.md5(new_filing_url.encode())
        name = str(name.hexdigest())
        path = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, name)
        if force:
            try:
                shutil.rmtree(path)
            except:
                pass
        if not os.path.exists(path):
            os.makedirs(path)
        hp = HighlightedParser(lsi_models_path = path)
        new_filing_text = hp.create_sentences(new_filing_url)
        if not fuzzy:
            if lsi:
                sentence = hp.lsi_search(new_filing_text, highlighted_text, min_ratio, num_search)
            else:
                sentence = hp.find(highlighted_text, new_filing_text, ignore_money=ignore_money, ignore_date=ignore_date)
        else:
            sentence = hp.fuzzy_find_one(highlighted_text, new_filing_text, min_ratio=min_ratio, depth=depth, min_sentence_length=min_sentence_length)

        if len(sentence) > 0:
            if fuzzy:
                return json.dumps({"new_highlighted_text": sentence[2], "ratio": sentence[1]})
            else:
                return json.dumps({"new_highlighted_text": sentence})
        else:
            return json.dumps({"new_highlighted_text": ""})


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Highlighted Tool")
    argparser.add_argument("--host", type=str, default="localhost", help="Host")
    argparser.add_argument("--port", type=int, default="8081", help="Port")
    args = argparser.parse_args()

    cherrypy.config.update({'server.socket_host': args.host})
    cherrypy.config.update({'server.socket_port': args.port})

    cherrypy.quickstart(HighlightedServer())
    cherrypy.engine.block()
