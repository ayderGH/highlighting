import cherrypy
from hlparser import HighlightedParser
import requests
import json

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
        ignore_money = data['ignore_money'] if 'ignore_money' in data.keys() else True
        ignore_date = data['ignore_date'] if 'ignore_date' in data.keys() else True

        new_filing_html = requests.get(new_filing_url)

        hp = HighlightedParser()

        new_filing_text = hp.get_processed_html(new_filing_html.text)

        matches = hp.find_text(highlighted_text, new_filing_text, ignore_money=ignore_money, ignore_date=ignore_date)
        if len(matches) > 0:
            return json.dumps({"new_highlighted_text": matches[0][0]})
        else:
            return json.dumps({"new_highlighted_text": ""})


if __name__ == "__main__":
    cherrypy.config.update({'server.socket_host': "127.0.0.1"})
    cherrypy.config.update({'server.socket_port': 8001})

    cherrypy.quickstart(HighlightedServer())
    cherrypy.engine.block()
