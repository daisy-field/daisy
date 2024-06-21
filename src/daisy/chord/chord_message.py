# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from enum import Enum
from uuid import uuid4

import numpy as np


class MessageType(Enum):
    """Indicates how an incoming messages should be processed."""

    LOOKUP_RES = 2
    LOOKUP_REQ = 3
    STABILIZE = 4
    NOTIFY = 5
    FED_MODEL = 6


class MessageOrigin(Enum):
    """Indicates the opration from which a message originated.
    Used to categorize lookup response messages.
    """

    JOIN = 1
    FIX_FINGERS = 2
    FED_PEERS_REQ = 3


class Chordmessage:
    """Class for Chord messages."""

    id: uuid4
    type: MessageType
    peer_tuple: tuple[int, tuple[str, int]]
    model: list[np.ndarray]
    sender: tuple[int, tuple[str, int]]
    timestamp: float
    origin: MessageOrigin

    def __init__(
        self,
        msg_type: MessageType,
        sender: tuple[int, tuple[str, int]],
        timestamp: float,
        request_id: uuid4 = None,
        origin: MessageOrigin = None,
        peer_tuple: tuple[int, tuple[str, int]] = None,
        model_params: list[np.ndarray] = None,
    ):
        """Creates a new Chordmessage.

        :param request_id: Message identifier
        :param msg_type: Type of message for processing in receive function.
        :param peer_tuple: ID and address of the peer sent whithin the Chordmessage.
        :param sender: Sender of the Chordmessage.
        :param timestamp: Timestamp of the Chordmessage. Indicates when it was created.
        :param origin: Origin of the Chordmessage.
        """
        self.id = request_id
        self.type = msg_type
        self.peer_tuple = peer_tuple
        self.model = model_params
        self.sender = sender
        self.timestamp = timestamp
        self.origin = origin
