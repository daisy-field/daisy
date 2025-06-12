#!/bin/bash

# === Konfiguration ===
NEW_PORT=2222
USER="tempuser"
PASSWORT="TempPasswort123!"
TIMELIMIT=30  # Sekunden für SSH-Verfügbarkeit

# === Neuen Benutzer erstellen ===

sudo useradd -m -s /bin/bash $USER
echo "$USER:$PASSWORT" | sudo chpasswd
sudo usermod -aG sudo $USER

# === SSH-Konfiguration sichern und anpassen ===
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

sudo sed -i "/^#\?Port /c\Port $NEW_PORT" /etc/ssh/sshd_config
sudo sed -i "/^#\?PasswordAuthentication /c\PasswordAuthentication yes" /etc/ssh/sshd_config
sudo sed -i "/^#\?PermitRootLogin /c\PermitRootLogin no" /etc/ssh/sshd_config

# ===  ===
sudo ufw allow $NEW_PORT/tcp

# === SSH-Dienst neu starten ===
sudo systemctl restart ssh

# === Information anzeigen ===
echo -e "SSH is now available!"
echo "Benutzer: $USER"
echo "Passwort: $PASSWORT"
echo "Port: $NEW_PORT"
echo "FOr $TIMELIMIT Seconds..."
echo

# === Warten und dann alles rückgängig machen ===
sleep $TIMELIMIT

echo "[!] Time is over"

# Benutzer löschen
sudo deluser --remove-home $USER

# SSH-Konfiguration zurücksetzen
sudo mv /etc/ssh/sshd_config.bak /etc/ssh/sshd_config

# Firewall-Port schließen
sudo ufw delete allow $NEW_PORT/tcp

# SSH-Dienst neu starten
sudo systemctl restart ssh

echo "SSH is closed."
