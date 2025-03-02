# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod

from daisy.chord.bittorrent import BitTorrentPeer
from daisy.federated_ids_components import FederatedOnlinePeer


class FederatedOnlinePeerToNetworkInterface(ABC):
    """This interface defines the functionality that should be implemented by every
    network topology which may be used with the FederatedOnlinePeer."""

    _federated_online_peer: FederatedOnlinePeer
    _bit_torrent_peer: BitTorrentPeer

    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_n_federated_peers_random_selection(self, n_peers: int = 5):
        """Find n peers for federated learning.
        Arbitrarily chosen default value is 5."""
        pass

    @abstractmethod
    def get_federated_modeldata(self, fed_peers: set[tuple[int, tuple[str, int]]]):
        """Request modeldata from one or more peers"""
        pass

    @abstractmethod
    def send_modeldata(self):
        """Share own modeldata with one or more other peers"""
        pass
