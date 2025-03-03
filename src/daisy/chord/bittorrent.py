# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import threading
from typing import Callable

from daisy.chord import ChordDHTPeer
from daisy.chord.federated_online_peer_interface import (
    FederatedOnlinePeerToNetworkInterface,
)


class BitTorrentPeer(FederatedOnlinePeerToNetworkInterface):
    """ """

    _bt_tracker_peer: ChordDHTPeer
    _bt_hash_func: Callable[[tuple[str, int]], int]
    _bt_addr: tuple[str, int]
    _bt_entry_addr: tuple[str, int]

    def __init__(
        self,
        dht_id: int,
        bt_addr: tuple[str, int],
    ) -> None:
        self._tracker_peer = ChordDHTPeer(
            peer_id=self._bt_hash_func(self._bt_addr),
            addr=self._bt_addr,  # TODO what parameter is peer_id supposed to be?
        )
        super().__init__()

    def setup(self):
        """ """
        # start/join dht
        thread = threading.Thread(
            target=self._tracker_peer.start, args=[self._bt_entry_addr], daemon=True
        )
        thread.start()
        raise NotImplementedError

    def cleanup(self):
        """ """
        # depart from dht, cleanup remaining stuff
        raise NotImplementedError

    def get_n_federated_peers_random_selection(
        self, **kwargs
    ) -> set[tuple[int, tuple[str, int]]]:
        """Find peers for federated learning
        :returns: set with ids and addresses of found peers
        """
        raise NotImplementedError

    def get_federated_modeldata(self, **kwargs):
        """Request modeldata from one or more peers"""
        raise NotImplementedError

    def send_modeldata(self):
        """Share own modeldata with one or more other peers"""
        raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
