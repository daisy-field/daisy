import subprocess

def open_temp_ssh():
    path = "./open_ssh_temp.sh"

    try:
        subprocess.run(["bash", path], check = True )
    except subprocess.CalledProcessError:
        print(f"SSH port can not open!")

if __name__ =="__main__":
    open_temp_ssh()