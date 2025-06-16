import paramiko

def connect_to_target(target_ip, attack_type):
    port = 2222                  
    username = "tempuser" #geendert zu vm user
    password = "TempPassword123!" #geendert zu vmuser

    if attack_type == "reverse_shell":
        # bei verzeichnessen / or \ ???
        attack = "mv ~/daisy/src/daisy/communication/ThatCouldBeMaleware.png ~/daisy/src/daisy/communication/YouWereAttacked.png"
    
    elif attack_type == "path":
        attack = "cd /.."
    
    else:
        raise ValueError

    # SSH-Client 
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # akzeptiert unbekannte Hosts automatisch

    try:
        client.connect(target_ip, port=port, username=username, password=password)

        # Beispiel: Befehl ausführen
        stdin, stdout, stderr = client.exec_command(attack)

        print("stdout:",stdout.read().decode())
        print("stderr:",stderr.read().decode())

    except Exception as e:
        print(e)
    finally:
        client.close()
        print("Connection closed.")

if __name__ =="__main__":
    connect_to_target("127.0.0.1", "reverse_shell")