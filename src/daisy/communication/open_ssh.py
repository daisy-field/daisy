import subprocess

def open_temp_ssh():
    path = "./setup_temp_ssh.sh"

    try:
        subprocess.run(["bash", path], check = True )
    except subprocess.CalledProcessError:
        print(f"SSS port can not open!")