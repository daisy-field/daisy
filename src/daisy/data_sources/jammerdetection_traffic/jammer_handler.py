# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import threading
import queue
import json
import pandas as pd
import websocket
import logging
from typing import Iterator

from daisy.data_sources import DataSource


# --- Neue WebSocket-Implementierung ohne await ---
class JammerWebSocketDataSource(DataSource):
    """
    Sammelt Daten von einem WebSocket-Server synchron ein, indem
    eingehende Nachrichten in eine Queue gepuffert werden.
    Über __iter__ kann man sie dann Stück für Stück abrufen.
    """

    _ws_app: websocket.WebSocketApp

    def __init__(
        self,
        url: str,
        name: str = "WebSocketDataSource",
        log_level: int = logging.INFO,
        auto_start: bool = False,
    ):
        super().__init__(name=name, log_level=log_level)
        self.url = url
        self._ws_app: websocket.WebSocketApp = None
        self._thread: threading.Thread = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running = threading.Event()

        # WebSocketApp mit Callbacks
        self._ws_app = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        if auto_start:
            self.open()

    def _on_open(self, ws):
        self._logger.info(f"✅ Verbunden mit WebSocket-Server: {self.url}")
        # Beispiel-Initialisierungsmeldungen:
        ws.send("debug:true")
        self._logger.info("📤 Gesendet: debug:true")
        ws.send("normalize:true")
        self._logger.info("📤 Gesendet: normalize:true")
        ws.send("state:start")
        self._logger.info("📤 Gesendet: state:start")
        self._running.set()

    def _on_message(self, ws, message: str):
        self._logger.debug(f"📥 Empfangene Roh-Daten: {message}")
        self._queue.put(message)

    def _on_error(self, ws, error):
        self._logger.error(f"⚠️ WebSocket-Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self._logger.info(
            f"❌ Verbindung geschlossen (code={close_status_code}): {close_msg}"
        )
        self._running.clear()

    def open(self):
        """Startet den Hintergrund-Thread mit run_forever()."""
        self._thread = threading.Thread(target=self._ws_app.run_forever, daemon=True)
        self._thread.start()
        # Warten, bis on_open feuert:
        if not self._running.wait(timeout=5):
            raise RuntimeError("WebSocket-Connection konnte nicht aufgebaut werden.")

    def close(self):
        """Schließt die Verbindung und wartet auf Thread-Ende."""
        self._ws_app.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._logger.info("WebSocket-Client sauber beendet.")

    def __iter__(self) -> Iterator[dict]:
        """
        Gibt JSON-geparste Nachrichten zurück. Blockiert, bis neue
        vorhanden sind, oder endet, sobald close() gerufen wurde.
        """
        while self._running.is_set() or not self._queue.empty():
            try:
                raw = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                yield json.loads(raw)["Message"]
            except json.JSONDecodeError:
                self._logger.warning(
                    f'⚠️ Nachricht: "{raw}" war kein gültiges JSON, wird übersprungen.'
                )
                continue


# --- Beispiel-Nutzung ---
if __name__ == "__main__":
    # Ziel-WebSocket
    WS_SERVER_URL = "ws://10.42.6.111:4501/Phy"

    # DataFrame für empfangene Daten
    df = pd.DataFrame()

    # DataSource öffnen
    src = JammerWebSocketDataSource(
        url=WS_SERVER_URL, log_level=logging.DEBUG, auto_start=True
    )

    try:
        for packet in src:
            # Hier hast du nun synchron ein Python-dict
            print("Empfangenes Paket:", packet)
            df = pd.concat([df, pd.DataFrame([packet])], ignore_index=True)
            print(df.tail(1))
    finally:
        # Sauber schließen
        src.close()
