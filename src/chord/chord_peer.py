"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23

"""
import argparse
import logging
import random
import threading
import typing
from enum import Enum
from time import time, sleep
from typing import Optional
from uuid import uuid4

import numpy as np

from chord.lib import *
from daisy.communication import StreamEndpoint, EndpointServer


class MessageOrigin(Enum):
    JOIN = 1
    FIX_FINGERS = 2
    TEST = 3


class MessageType(Enum):
    JOIN = 1
    LOOKUP_SUCC_RES = 2
    LOOKUP_SUCC_REQ = 3
    STABILIZE = 4
    NOTIFY = 5
    TEST = 6


class Chordmessage:
    """Class for Chord messages.
    """
    message_id: uuid4
    message_type: MessageType
    peer_tuple: tuple[int, tuple[str, int]]
    success: bool

    def __init__(self, message_id: uuid4, message_type: MessageType, peer_tuple: tuple[int, tuple[str, int]] = None):
        """Creates a new Chordmessage.
        :param message_id: Message identifier
        :param message_type: Type of message for processing in receive function.
        :param peer_tuple: Id and address of the peer sent whithin the Chordmessage.
        """
        self.message_id = message_id
        self.message_type = message_type
        self.peer_tuple = peer_tuple


def close_tmp_ep(ep: StreamEndpoint, sleep_time: int = 10, stop_timeout: int = 10):
    sleep(sleep_time)
    ep.stop(shutdown=True, timeout=stop_timeout)


class Chordpeer:
    # TODO: have succ in finger table and not external
    """Class for Chordpeers.
    """

    _id: int
    _name: str  # for debugging and fun
    _addr: tuple[str, int]

    _fingertable: dict[int, tuple[int, tuple[str, int], StreamEndpoint]]
    _successor: tuple[int, tuple[str, int]] | None
    _successor_endpoint: StreamEndpoint | None
    _predecessor: tuple[int, tuple[str, int]] | None
    _predecessor_endpoint: StreamEndpoint | None

    _endpoint_server: EndpointServer
    _max_fingers: int  # let m/max_fingers be the number of bits in the key/peer identifiers. (copied from paper)

    _sent_messages: dict[uuid4, tuple[MessageOrigin, time, Optional[int]]]

    _logger: logging.Logger
    _refresh_interval: float

    def __init__(self, name: str, addr: tuple[str, int], max_fingers: int = 32, id_test: int = None,
                 refresh_interval: float = 5.0):
        """
        Creates a new Chord Peer.

        :param name: Name of peer for logging and fun.
        :param addr: Address of peer.
        :param max_fingers: Max size of fingertable, also used for calculating peer ids on chord
        """

        self._id = id_test  # hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr

        self._fingertable = {}
        self._successor = None  # vllt. doch auf self initen?
        self._successor_endpoint = None
        self._predecessor = (self._id, self._addr)
        self._predecessor_endpoint = None

        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True, c_timeout=30)
        self._max_fingers = max_fingers

        self._refresh_interval = refresh_interval
        self._sent_messages = {}

        self._logger = logging.getLogger(name)

    def _send_message(self, ep: StreamEndpoint | None, remote_addr: tuple[str, int], message: Chordmessage):
        """Sends a Chordmessage over a given endpoint or temporarily creates a new one.
        TODO Initiates clean up if the given endpoint is no longer usable, but sends the message nevertheless.

        Note: The given Endpoint may be None, or inactive. The peer to whom the Message should be sent may have failed,
        or left the chordring. Due to the asychronicity of the Endpoints sending message and immediately closing
        the endpoint may lead to losing the message.

        :param ep: Endpoint to a receiving Chord peer.
        :param remote_addr: Address of a Chord peer
        :param message: Chordmessage to be sent
        """
        try:
            ep.send(message)
            self._logger.info(
                f"Sent Message of Type {message.message_type} From {self._name, self._id} To {remote_addr} With {message.peer_tuple}")
        except (AttributeError, RuntimeError) as e:  # attributeError if ep is none, runtimeError if ep is not started
            self._logger.warning(f"{e.__class__.__name__} ({e}) :: in _send_message: creating new ep to send.")

            ep_name = f"one-time-ep-{np.random.randint(0, 100)}"
            endpoint = StreamEndpoint(name=ep_name, remote_addr=remote_addr, acceptor=False, multithreading=True,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(message)
            self._logger.info(
                f"Sent Message of Type {message.message_type} From {self._name, self._id} To {remote_addr} With {message.peer_tuple}")
            threading.Thread(target=lambda: close_tmp_ep(endpoint, 10, 10), daemon=True).start()

    def _join(self, remote_addr: tuple[str, int]):
        """Creates and sends a join message to a peer in an existing chord ring.

        # TODO retry after ttl of message runs out -> check ttl of messages periodically?

        :param remote_addr: Address of bootstrap peer.
        """

        self._logger.info("JOINKING: " + self.__str__())

        message_id = uuid4()
        self._sent_messages[message_id] = (MessageOrigin.JOIN, time(), None)
        join_chord_message = Chordmessage(message_id=message_id, message_type=MessageType.JOIN,
                                          peer_tuple=(self._id, self._addr))
        self._send_message(None, remote_addr, join_chord_message)

    def cleanup_dead_messages(self):
        """Checks whether sent messages can be forgotten and deletes them.
        """
        for key in list(self._sent_messages):
            if self._sent_messages.get(key, None) is not None:
                if time() - self._sent_messages[key][1] > self._refresh_interval:
                    self._sent_messages.pop(key)

    def _notify(self, remote_id: int, remote_addr: tuple[str, int]):
        """Creates and sends a notify message to validate a peers predecessor. Used to repair successor pointers on the chord ring.
        """
        if self._predecessor[0] == self._id:
            self._logger.warning(f"Notify to self")
            return
        notify_message = Chordmessage(message_id=uuid4(), message_type=MessageType.NOTIFY, peer_tuple=self._predecessor)
        ep = self._get_existing_or_new_endpoint(remote_id, remote_addr)
        self._send_message(ep, remote_addr, notify_message)

    def _stabilize(self, remote_id: int, remote_addr: tuple[str, int]):
        """Creates and sends a stabilize message to the sucessor of a peer, initiating reparation of its successor and respective predecessor pointers.

        :param remote_id: Chord-Id of successor
        :param remote_addr: Address of successor
        """
        if (remote_id is None) or (remote_addr is None):
            self._logger.warning(f"Stabilize missing value (remote_id:{remote_id}, remote_addr:{remote_addr})")
            return
        # if remote_id == self._id:
        #    self._logger.warning(f"Stabilize to self")
        #    return
        stabilize_message = Chordmessage(message_id=uuid4(), message_type=MessageType.STABILIZE,
                                         peer_tuple=(self._id, self._addr))
        ep = self._get_existing_or_new_endpoint(remote_id, remote_addr)
        self._send_message(ep, remote_addr, stabilize_message)

    def _update_finger(self, finger_tuple: tuple[int, tuple[str, int]], finger_index: int):
        """Updates an entry in a peers fingertable by maintaining its endpoint and assigning current finger-values or
        removing the finger entirely.

        :param finger_tuple:
        :param finger_index:
        :return:
        """
        finger = self._fingertable.pop(finger_index, None)
        if finger is not None:
            finger_ids_in_ft = {finger_id for finger_id, finger_addr, finger_ep in self._fingertable.values() if
                                finger[2] is finger_ep}
            if len(finger_ids_in_ft) == 0:
                if finger[2].poll()[0][0]:
                    finger[2].stop(shutdown=True)
        try:
            ep = self._get_existing_or_new_endpoint(*finger_tuple)
            self._fingertable[finger_index] = (finger_tuple[0], finger_tuple[1], ep)
        except TypeError:
            pass

    def _get_existing_or_new_endpoint(self, remote_id: int, remote_addr: tuple[str, int]) -> Optional[StreamEndpoint]:
        """Checks whether a endpoint to a given peer exists and returns it. Returns fresh endpoint if none
        exists.

        Note: Endpoints, or peers, may fail at any point within or outside the function. Any endpoint returned by this
        function may be dead. This problem remains to be solved somehow.

        :param remote_id: peer id to check connection to
        :param remote_addr: address of remote peer
        :return ep: StreamEndpoint to peer or None
        """
        if self._predecessor_endpoint is not None and remote_id == self._predecessor[0]:
            return self._predecessor_endpoint
        if self._successor_endpoint is not None and remote_id == self._successor[0]:
            return self._successor_endpoint

        # ep_as_set = {finger_ep for finger_id, finger_addr, finger_ep in self._fingertable.values()
        # if remote_id == finger_id}
        # try:
        #    return ep_as_set.pop()
        # except KeyError as e:
        #    self._logger.error(f"{e.__class__.__name__} ({e}) :: in get_ep_if-exists: No ep found. Creating new ep")

        ep = StreamEndpoint(name=f"get-ep-{random.randint(0, 100)}", remote_addr=remote_addr, acceptor=False,
                            multithreading=True, buffer_size=10000)
        ep.start()
        return ep

    def _fix_fingers(self):
        """Iterates the finger indices and initiates lookups for the respective peers; defaults to the successor.

        Note: this method does not actually update any finger table entries. It will return early if a peer has no successor.
        """
        # if self._successor is None:
        self._logger.info(f"Starting to update Fingers... ")
        # return

        for i in range(self._max_fingers):
            finger = self._id + 2 ** i % (2 ** self._max_fingers)
            if check_if_peer_is_successor(self._successor[0], self._id, finger):  # fixme why this?
                print("updateing finger with successor")
                self._update_finger(self._successor, i)
            else:
                message_id = uuid4()
                self._sent_messages[message_id] = (MessageOrigin.FIX_FINGERS, time(), i)
                print("making new lookup succ call with ", finger)
                self._lookup_successor(finger, self._addr,
                                       message_id)

    def _get_closest_known_pred(self, peer_id: int) -> tuple[int, tuple[str, int], StreamEndpoint] | None:
        """Finds the closest predecessor one peer knows for another peer in the fingertable. Defaults to None.

        :param peer_id:
        :return: closest predecessor from fingertabel or None
        """
        if len(self._fingertable) == 0:
            self._logger.info(f"In _get_closest_known_pred: empty fingertable; returning None")
            return None

        for i in range(0, self._max_fingers, -1):
            try:
                finger = self._fingertable[i]
                if check_if_peer_is_between(self._id, finger[0], peer_id):
                    self._logger.info(f"In _get_closest_known_pred: returning finger {finger[0]}")
                    return finger
            except TypeError as e:
                self._logger.error(f"{e.__class__.__name__} ({e}) :: in _get_closest_known_pred")
            except ValueError as e:
                self._logger.error(f"{e.__class__.__name__} ({e}) :: in _get_closest_known_pred")
        self._logger.info(f"In _get_closest_known_pred: returning None")

    def _lookup_successor(self, peer_id: int, peer_addr: tuple[str, int], message_id: uuid4):
        """Checks whether a peer knows the sucessor of the given peer. If it knows the successor it will send back its
        (id, addr) pair, otherwise the next peer will be asked.

        :param peer_addr: chord address of peer who initially sent the find_successor request
        :param peer_id: chord id of peer whose successor should be found
        :message_id:
        """
        self._logger.info(f"looking up {peer_id}.")

        # second condition: if two peers are in ring A should not send B to B as its successor
        if peer_addr == self._predecessor[1] or check_if_peer_is_predecessor(self._predecessor[0], self._id, peer_id):
            self._logger.info(f"In _find_succ: found succ for {peer_id} as self ")
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.LOOKUP_SUCC_RES,
                                          peer_tuple=(self._id, self._addr))
            ep = self._get_existing_or_new_endpoint(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)

        elif check_if_peer_is_successor(self._successor[0], self._id, peer_id):
            self._logger.info(f"In _find_succ: found succ for {peer_id} as self._successor:{self._successor[0]} ")
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.LOOKUP_SUCC_RES,
                                          peer_tuple=self._successor)
            ep = self._get_existing_or_new_endpoint(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)  # not closing ep if it comes from check

        else:
            self._logger.info(f"In _find_succ: no succ found for {peer_id}")

            # send negative response including the closest predecessor from fingertable
            try:
                # closest_pred = self._get_closest_known_pred(peer_id)
                if self._successor[0] != self._id:
                    self._successor_endpoint.send(
                        Chordmessage(message_id=message_id, message_type=MessageType.LOOKUP_SUCC_REQ,
                                     peer_tuple=(peer_id, peer_addr)))
                # print("hello")
            except AttributeError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in find_succ: failed to relay lookup; no ep for closest pred "
                    f"found in ft")
                succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.LOOKUP_SUCC_RES,
                                              peer_tuple=(self._id, self._addr))
                ep = self._get_existing_or_new_endpoint(peer_id, peer_addr)
                self._send_message(ep, peer_addr, succ_found_msg)

    def _process_join(self, message: Chordmessage):
        """Handles incoming join messages. Assigns the successor of self if it is None
        and initiates successor lookup for new peer.

        :param message:
        :return:
        """

        if self._successor is None:
            self._logger.info(f"In _process_join: updating successor from None to {self._successor}.")
            self._successor = message.peer_tuple
            self._successor_endpoint = StreamEndpoint(name=f"succ-ep-{self._name}", remote_addr=message.peer_tuple[1],
                                                      acceptor=False, multithreading=True, buffer_size=10000)
            self._successor_endpoint.start()
        self._logger.info(f"In _process_join: initiating _find_succ for {message.peer_tuple}.")
        self._lookup_successor(*message.peer_tuple, message.message_id)

    def _process_notify(self, message: Chordmessage):
        """Handles incoming notify messages.

        :param message:
        :return:
        """
        if message.peer_tuple[0] == self._id:
            return
        if check_if_peer_is_successor(self._successor[0], self._id, message.peer_tuple[0]):
            self._logger.info(f"In _process_notify: updating successor from {self._successor} to {message.peer_tuple}.")
            self._successor = message.peer_tuple
            ep = self._get_existing_or_new_endpoint(*message.peer_tuple)
            if ep is None:
                self._successor_endpoint = StreamEndpoint(name=f"{self._name}-successor-ep-{np.random.randint(0, 100)}",
                                                          remote_addr=self._addr, acceptor=False, multithreading=True,
                                                          buffer_size=10000)
                self._successor_endpoint.start()
            else:
                self._successor_endpoint = ep

    def _process_stabilize(self, message: Chordmessage):
        """Handles incoming stabilize messages. Updates the predecessor of a peer or returns.

        :param message: Contains the id of a peer who believes the receiving peer is its successor.
        """
        # if pred is still self change!!
        if self._predecessor == (self._id, self._addr):
            self._predecessor = message.peer_tuple
        # if (peer id is my pred (I am succ) or between me and my pred) update my pred
        elif check_if_peer_is_predecessor(self._predecessor[0], self._id, message.peer_tuple[0]) and (
                self._id != message.peer_tuple[0]):
            self._predecessor = message.peer_tuple
            ep = self._get_existing_or_new_endpoint(*message.peer_tuple)
            if ep is None:
                self._predecessor_endpoint = StreamEndpoint(
                    name=f"{self._name}-predecessor-ep-{np.random.randint(0, 100)}", remote_addr=self._addr,
                    acceptor=False, multithreading=True, buffer_size=10000)
            else:
                self._predecessor_endpoint = ep
        self._notify(*message.peer_tuple)

    def _process_lookup_succ_req(self, message: Chordmessage):
        """Handles incoming find successor request messages. Terminates the search if the request contains self, or initiates new find successor step.

        :param message: received Chordmessage
        """
        # call find succ with peer from message as target
        if message.peer_tuple[0] == self._id:
            self._logger.info(f"In _process_find_succ_req: terminating search because find_succ on self.")
            return
        self._logger.info(f"Init _find_succ for {message.peer_tuple[0]}, In _process_find_succ_req.")
        self._lookup_successor(*message.peer_tuple, message.message_id)

    def _process_lookup_succ_res(self, message: Chordmessage):
        """Handles incoming find successor response messages. Updates the successor of a peer and initiates stabilize,
        updates a finger in the fingertable, depending on the message origin.

        :param message: received Chordmessage
        :return:
        """
        if message.message_id not in self._sent_messages:
            self._logger.info(f"Received dead Message {message.peer_tuple}, In _process_find_succ_res.")
            return

        msg_origin, _, msg_finger = self._sent_messages.pop(message.message_id, (None, None, None))
        self._logger.info(f"Received {message.peer_tuple}, {msg_origin}, In _process_find_succ_res.")

        if msg_origin == MessageOrigin.JOIN:
            self._successor = message.peer_tuple
            self._successor_endpoint = StreamEndpoint(name=f"{self._name}-sucessor-ep-{np.random.randint(0, 100)}",
                                                      remote_addr=self._addr, acceptor=False, multithreading=True,
                                                      buffer_size=10000)
            self._successor_endpoint.start()
            self._logger.info(f"Updated successor to {self._successor}. In _process_find_succ_res.")
            self._stabilize(*message.peer_tuple)
        if msg_origin == MessageOrigin.FIX_FINGERS:
            self._logger.info(f"Initiating _update_finger for {msg_finger}. In _process_find_succ_res.")
            self._update_finger(message.peer_tuple, msg_finger)

    def run(self, bootstrap_addr: tuple[str, int] = None):
        """Starts the chord peer, joining an existing chord ring or bootstrapping a new one, before starting the periodic
        loop to maintain the ring and process incoming messages.

        :param self_id:
        :param bootstrap_addr: Remote address of peer to join existing ring, None to start new ring.
        """
        last_refresh_time = time()
        self._endpoint_server.start()

        if bootstrap_addr is not None:
            self._logger.info(f"Bootstrapaddr: {bootstrap_addr}")
            self._join(bootstrap_addr)

        while True:
            # check here for departure or not
            curr_time = time()
            if curr_time - last_refresh_time >= self._refresh_interval:
                try:
                    self._stabilize(*self._successor)
                    self._fix_fingers()
                except TypeError as e:
                    self._logger.error(f"{e.__class__.__name__} ({e}) :: could not Stabilize")
                cleanup_dead_messages(self._sent_messages, self._refresh_interval * 2)
                last_refresh_time = curr_time

            sleep(1)

            received_messages = self._receive_on_endpoints()
            for message in received_messages:
                msg_type = message.message_type
                self._logger.info(f"In run: Received {msg_type.name} with {message.peer_tuple}")
                match msg_type:
                    case MessageType.JOIN:
                        self._process_join(message)
                        self._logger.info("JOINK Processed: " + self.__str__())
                    case MessageType.LOOKUP_SUCC_RES:
                        self._process_lookup_succ_res(message)
                        self._logger.info("FIND_SUCC_RES Processed: " + self.__str__())
                    case MessageType.LOOKUP_SUCC_REQ:
                        self._process_lookup_succ_req(message)
                        self._logger.info("FIND_SUCC_REQ Processed: " + self.__str__())
                    case MessageType.STABILIZE:
                        self._process_stabilize(message)
                        self._logger.info("STABYOULIZE Processed: " + self.__str__())
                    case MessageType.NOTIFY:
                        self._process_notify(message)
                        self._logger.info(
                            "NOTIFLY Processed: " + self.__str__())  # self._logger.info(f"{msg_type.name} Processed: " + self.__str__())

    def _receive_on_endpoints(self) -> list[Chordmessage]:
        """
        Acquires a set of endpoints and receives all messages from their buffers.

        :return: List of received Chordmessages
        """
        endpoints = get_readable_endpoints(self._fingertable.values(), self._successor_endpoint,
                                           self._predecessor_endpoint, self._endpoint_server)
        messages = []
        for endpoint in endpoints:
            try:
                while True:
                    messages.append(typing.cast(Chordmessage, endpoint.receive(timeout=0)))
            except (RuntimeError, TimeoutError):
                pass
        return messages

    def __str__(self):
        return (
                f"ChordPeer {self._name}:\n" + f"id: {self._id}, \n" + f"predecessor: {self._predecessor}, \n" + f"successor: {self._successor}, \n" + f"fingers: {fingertable_to_string(self._fingertable)}, \n" + f"currently waiting for responses to {len(self._sent_messages)} messages.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--peerName", help="Peer name")
    parser.add_argument("--peerPort", type=int, default=None, help="Port of peer")
    parser.add_argument("--remotePort", type=int, default=None, help="Port of remote peer")
    parser.add_argument("--peerId", type=int, default=None, help="Id of remote peer")
    args = parser.parse_args()

    peer_ip = "127.0.0.1"

    peer = Chordpeer(name=args.peerName, addr=(peer_ip, args.peerPort), max_fingers=10, id_test=args.peerId)
    if args.remotePort is None:
        peer.run()  # start as first chord peer
    else:
        peer.run(bootstrap_addr=(peer_ip, args.remotePort))  # join existing chord
