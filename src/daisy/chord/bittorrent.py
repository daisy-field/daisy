# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from daisy.chord import Peer
from daisy.federated_ids_components import FederatedOnlinePeer


class FederatedOnlineBitTorrentPeer(FederatedOnlinePeer):
    """ """

    _topology: Peer

    def setup(self):
        """ """
        raise NotImplementedError

    def cleanup(self):
        """ """
        raise NotImplementedError

    def create_async_fed_learner(self):
        """ """
        raise NotImplementedError
