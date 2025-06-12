import paramiko


def connect_to_taget(taget_ip, attack_type):
    port = 2222                  
    username = "tempuser"
    password = "TempPassword123!"

    if attack_type == "revers_shell":
        # bei verzeichnessen / or \ ???
        attack = "mv /daisy/src/daisy/communcation/ThatCouldBeMaleware.png /daisy/src/daisy/communcation/YouWareAttacked.png"
    
    elif attack_type == "parth":
        attack = "cd //"

    # SSH-Client 
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # akzeptiert unbekannte Hosts automatisch

    try:
        client.connect(taget_ip, port=port, username=username, password=password)

        # Beispiel: Befehl ausführen
        stdin, stdout, stderr = client.exec_command(attack)

        print(stdout.read().decode())

    except Exception as e:
        print(f"Connection failed")
    finally:
        client.close()
        print("🔒 Verbindung geschlossen.")
