"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23

"""
import datetime
import logging
import threading
from time import time

from uuid import uuid4
from enum import Enum
from typing import Optional
import numpy as np
import typing

from src.communication.message_stream import StreamEndpoint, EndpointServer


class MessageType(Enum):
    JOIN = 1
    FIND_SUCC_RES = 2
    FIND_SUCC_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class MessageOrigin(Enum):
    JOIN = 1
    FIX_FINGERS = 2


class Chordmessage:
    """Class for Chord messages.
    """
    message_id: uuid4
    message_type: MessageType
    peer_tuple: tuple[int, tuple[str, int]]

    def __init__(self, message_id: uuid4, message_type: MessageType,
                 peer_tuple: tuple[int, tuple[str, int]] = None):
        """Creates a new Chordmessage.
        :param message_id: Message identifier
        :param message_type: Type of message for processing in receive function.
        :param peer_tuple: Id and address of the peer sent whithin the Chordmessage.
        """
        self.message_id = message_id
        self.message_type = message_type
        self.peer_tuple = peer_tuple


class Chordpeer:
    # TODO HIGH PRIO: REFACTOR EP handling

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

    def __init__(self, name: str, addr: tuple[str, int],
                 max_fingers: int = 32):
        """
        Creates a new Chord Peer.

        :param name: Name of peer for logging and fun.
        :param addr: Address of peer.
        :param max_fingers: Max size of fingertable, also used for calculating peer ids on chord
        """

        self._id = hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr

        self._fingertable = {}
        self._successor = None  # vllt. doch auf self initen?
        self._successor_endpoint = None
        self._predecessor = (self._id, self._addr)
        self._predecessor_endpoint = None

        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True)
        self._max_fingers = max_fingers

        self._sent_messages = {}

        self._logger = logging.getLogger(name + "-Peer")

    def _send_message(self, ep, remote_addr, message):
        """Sends a Chordmessage over a given endpoint. Initiates clean up if the given endpoint is no longer usable, but sends the message nevertheless.

        Note: The given Endpoint may be None, or inactive. The Node to whom the Message should be sent may have failed,
        or left the chordring. Due to the sychronicity of the Endpoints sending message and immediately closing
        the endpoint may lead to losing the message.

        :param ep: Endpoint to a receiving Chord Node.
        :param remote_addr: Address of a Chord Node
        :param message: Chordmessage to be sent
        """
        if ep is not None:
            # if self._clean_up_ep_if_dropout(ep):
            #    return
            ep.send(message)
            self._logger.info(f"Sent Message of Type {message.message_type} with Peer {message.peer_tuple}")
        else:
            ep_name = f"one-time-ep-{np.random.randint(0, 100)}"
            endpoint = StreamEndpoint(name=ep_name,
                                      remote_addr=remote_addr, acceptor=False, multithreading=True,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(message)
            self._logger.info(
                f"Sent Message of Type {message.message_type} with Peer {message.peer_tuple} via {ep_name}")
            threading.Thread(target=lambda: endpoint.stop(shutdown=True, timeout=30), daemon=True).start()

    def _join(self, remote_addr: tuple[str, int]):
        """Creates and sends a join message to a peer in an existing chord ring.

        # TODO if message is lost should be retried after ttl of message runs out -> method to check ttl of messages
        periodically?

        :param remote_addr: Address of bootstrap peer.
        """
        message_id = uuid4()
        self._sent_messages[message_id] = (MessageOrigin.JOIN, time(), None)
        join_chord_message = Chordmessage(message_id=message_id, message_type=MessageType.JOIN,
                                          peer_tuple=(self._id, self._addr))
        self._send_message(None, remote_addr, join_chord_message)

    def check_sent_messages_ttl(self):
        """Checks whether sent messages can be forgotten and deletes them.
        """
        for key in self._sent_messages.values():
            if self._sent_messages[key][1] is not None:
                if time() - self._sent_messages[key][1] > 60:
                    self._sent_messages.pop(key)

    def _notify(self, remote_id: int, remote_addr: tuple[str, int]):
        """Send the predecessor of a peer to another peer. Used to repair successor pointers on the cord ring.

        :param remote_id:
        :param remote_addr:
        """
        notify_message = Chordmessage(message_id=uuid4(), message_type=MessageType.NOTIFY,
                                      peer_tuple=self._predecessor)
        ep = self._get_ep_if_exists(remote_id, remote_addr)
        self._send_message(ep, remote_addr, notify_message)

    def _stabilize(self, remote_id: int, remote_addr: tuple[str, int]):
        """Sends self to successor, initiating reparation of successor and predecessor pointers.

        TODO Succ may have failed in the meantime, ask next fingertable entry or predecessor
        Fixme: succ ist self

        :param remote_id: Chord-Id of successor
        :param remote_addr: Address of successor
        """
        if remote_id is None or remote_addr is None:
            self._logger.warning(f"MISSING VALUE in STABILIZE (remote_id:{remote_id}, remote_addr:{remote_addr})")
            return
        stabilize_message = Chordmessage(message_id=uuid4(), message_type=MessageType.STABILIZE,
                                         peer_tuple=(self._id, self._addr))
        ep = self._get_ep_if_exists(remote_id, remote_addr)
        self._send_message(ep, remote_addr, stabilize_message)

    def _update_finger(self, finger_peer: tuple[int, tuple[str, int]], finger_index: int):
        """Updates an entry in a peers fingertable by maintaining its StreamEndpoint and assigning current finger-values or removing the finger entirely.

        :param finger_peer:
        :param finger_index:
        :return:
        """
        finger = self._fingertable.pop(finger_index, None)
        if finger is not None:
            finger_ids_in_ft = {finger_id for finger_id, finger_addr, finger_ep in self._fingertable.values()
                                if finger[2] is finger_ep}
            if len(finger_ids_in_ft) == 0:
                self._close_and_remove_ep(finger[1], finger[2])

        ep = self._get_ep_if_exists(*finger_peer)  # checks ep-server and self._finger_table, in case ep has not been
        # removed already
        # if ep is not None:
        # if self._clean_up_ep_if_dropout(ep):
        # return
        # else:  # create new ep if none exists
        if ep is None:
            ep = StreamEndpoint(name=f"finger-ep-{finger_index}",
                                remote_addr=finger_peer[1], acceptor=False, multithreading=True,
                                buffer_size=10000)
            ep.start()
        self._fingertable[finger_index] = (finger_peer[0], finger_peer[1], ep)

    def _get_ep_if_exists(self, remote_id: int, remote_addr: tuple[str, int]) -> Optional[StreamEndpoint]:
        """Checks whether a StreamEndpoint to a given peer exists and returns it. Returns None if StreamEndpoint does not exist.

        Note: Endpoints, or peers, may fail at any point within or outside the function. Any endpoint returned by this function may be dead. This problem remains to be solved somehow.

        :param remote_id: peer id to check connection to
        :param remote_addr: address of remote peer
        :return ep: StreamEndpoint to peer or None
        """
        if remote_id == self._predecessor[0] and self._predecessor_endpoint is not None:
            return self._predecessor_endpoint
        if remote_id == self._successor[0] and self._successor_endpoint is not None:
            return self._successor_endpoint

        ep = {finger_ep for finger_id, finger_addr, finger_ep in self._fingertable.values()
              if remote_id == finger_id}
        try:
            return ep.pop()
        except KeyError as e:
            self._logger.error(f"{e.__class__.__name__} ({e}) :: in get_ep_if-exists: No ep found")
            return

    def _clean_up_ep_if_dropout(self, ep: StreamEndpoint) -> bool:  # currently not in useage
        """Searches through all available connections of a peer to see where a given StreamEndpoint is used and initiates cleanup of the StreamEndpoint if it has died.
        FIXME how do i actually know if ep is dropout? send ping?
        TODO predecessor und successor updates auslagern
        Note: may result in removal of successor or predecessor endpoint of a peer or empty fingertables.

        :param ep: StreamEndpoint suspected as dropout
        """
        if ep is None:  # dropout
            return True

        states, addrs = ep.poll()  # supposed to filter eps that are not dropouts, everything after this line assumes dropout ep
        if states[0] or states[1] or states[2]:  # may not work like this, ep can be not dropout but unavailable
            return False

        dropouts = [finger_key for finger_key, finger in self._fingertable.items()
                    if ep is finger[2]]
        for do in dropouts:
            do_id, do_addr, do_ep = self._fingertable.pop(do, (None, None, None))
            if do_ep is None:
                continue
            self._close_and_remove_ep(do_addr, do_ep)  # close finger_eps that are ep

        if ep is self._predecessor_endpoint:  # clean up predecessor ep if necessary
            self._close_and_remove_ep(self._predecessor[1], ep)
            self._predecessor = None
            self._predecessor_endpoint = None
        if ep is self._successor_endpoint:  # clean up succ ep if necessary
            self._close_and_remove_ep(self._successor[1], ep)
            for finger in range(
                    self._max_fingers):  # set some arbitrary finger as succ, will be repaired by stabilize/notify calls
                f_id, f_addr, f_ep = self._fingertable.get(finger, (None, None, None))
                if not self._clean_up_ep_if_dropout(f_ep):
                    self._successor = (f_id, f_addr)
                    self._successor_endpoint = f_ep
                    break
            if self._successor is None:
                self._successor = self._predecessor
                self._successor_endpoint = self._predecessor_endpoint
        return True

    def _close_and_remove_ep(self, ep_addr: tuple[str, int], ep: StreamEndpoint):
        """
        :param ep_addr: Address of StreamEndpoint to remove
        :param ep: StreamEndpoint to remove
        """
        ep_in_epserver = self._endpoint_server.get_connections([ep_addr]).get(ep_addr)
        if ep_in_epserver is None:
            if ep.poll()[0][0] and ep is not None:
                return
                # ep.stop(shutdown=True)
        # else:
        #   self._endpoint_server.close_connections([ep_addr])

    def _fix_fingers(self):
        """Iterates the finger indices, initiating lookups for the respective peers; defaults to the successor.

        Note: this method does not actually update any finger table entries. It will return early if a node has no successor.
        """
        # if self._successor is None:
        # self._logger.warning(f"In fix_fingers: Successor is None, ending fix_fingers. ")
        # return

        for i in range(self._max_fingers):
            finger = self._id + 2 ** i % (2 ** self._max_fingers)
            if self._is_pred(finger):
                self._update_finger(self._successor, i)
            else:
                message_id = uuid4()
                self._sent_messages[message_id] = (MessageOrigin.FIX_FINGERS, time(), i)
                self._find_succ(finger, self._addr, message_id)

    def _is_succ(self, peer_id: int) -> bool:
        """Checks whether self is the successor of a peer or not. Successor may be None.

        Note: will return True if s
        :param peer_id:
        :return: true if self is successor of peer, else false
        """
        if self._predecessor[0] == self._id:
            return True
        return (self._id < self._predecessor[0]) & (peer_id not in range(self._id + 1, self._predecessor[0] + 1)) \
            or (peer_id in range(self._predecessor[0] + 1, self._id + 1))

    def _is_pred(self, peer_id: int) -> bool:
        """Checks whether self is predecessor of a peer or not. Predecessor may be None. Throws TypeError.

        :param peer_id:
        :return: true if self is predecessor of peer, else false
        """
        # if peer_id == self._id == self._successor[0]:
        #  return False

        try:
            return (self._id > self._successor[0]) & (peer_id not in range(self._successor[0] + 1, self._id + 1)) \
                or (peer_id in range(self._id + 1, self._successor[0] + 1))
        except TypeError as e:
            self._logger.error(f"{e.__class__.__name__} ({e}) ::  in _is_pred")
            return False

    def _get_closest_known_pred(self, peer_id: int) -> StreamEndpoint | None:
        """Finds the closest predecessor one peer knows for another peer in fingertable.

        # FIXME only one finger - should work now
        Note: If the fingertable is empty throws an Error and returns None.

        :param peer_id:
        :return:
        """
        if self._successor is None or (self._successor[0] == self._id):
            if self._predecessor is None or (self._predecessor[0] == self._id):
                if len(self._fingertable) == 0:
                    self._logger.warning(
                        f"In function self._get_closest_known_pred: Peer is not connected in Chord; Pred and succ are None or self, empty Fingertable")
                    return

        for f_index in range(self._max_fingers):
            f_curr = self._fingertable.get(f_index, None)  # id, addr, ep
            f_next = self._fingertable.get(f_index + 1, None)

            i = f_index + 1
            while (f_curr is None or f_next is None) and i < self._max_fingers:
                if f_curr is None:
                    f_curr = self._fingertable.get(i, None)
                if f_next is None:
                    f_next = self._fingertable.get(i + 1, None)
                i += 1
                if f_curr[0] == f_next[0] and f_curr is not None:
                    f_next = self._fingertable.get(i, None)

            # fingertable has only one entry, so either curr or next must be none
            if f_curr is None and f_next is not None:
                return f_next[2]
            elif f_next is None and f_curr is not None:
                return f_curr[2]

            try:
                if (f_curr[0] < f_next[0]) & (peer_id in range(f_curr[0], f_next[0] + 1)):
                    return f_curr[2]
                if (f_curr[0] > f_next[0]) & (peer_id in range(f_next[0], f_curr[0] + 1)):
                    return f_next[2]
            except TypeError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in _get_closest_known_pred: Maybe Fingertable of Peer is empty?")
                return
        return

    def _find_succ(self, peer_id: int, peer_addr: tuple[str, int], message_id: uuid4):
        """Function to find successor of peer with chord id find_succ_id. Message will be relayed along the cordring
        until the successor is found. In the end, the chord id and chord address of the found peer will be sent back
        directly to the peer who initially send the request.

        :param peer_addr: chord address of peer who initially sent the find_successor request
        :param peer_id: chord id of peer whose successor should be found
        :message_id:
        """

        if peer_id == self._id:
            return

        if self._is_pred(peer_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          peer_tuple=self._successor)
            ep = self._get_ep_if_exists(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)  # not closing ep if it comes from check

        elif self._is_succ(peer_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          peer_tuple=(self._id, self._addr))
            ep = self._get_ep_if_exists(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)
        # idk, ask closest pred of peer
        else:
            closest_pred_ep = self._get_closest_known_pred(peer_id)
            try:
                closest_pred_ep.send(Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_REQ,
                                                  peer_tuple=(peer_id, peer_addr)))
            except AttributeError:
                self._logger.error(
                    f"AttributeError in find_succ: Maybe Peer is alone in/disconnected from Chord? Sending back self.")
                succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                              peer_tuple=(self._id, self._addr))
                ep = self._get_ep_if_exists(peer_id, peer_addr)
                self._send_message(ep, peer_addr, succ_found_msg)

    def _process_join(self, message: Chordmessage):
        """

        :param message:
        :return:
        """

        if self._successor is None:
            self._successor = message.peer_tuple
            self._successor_endpoint = StreamEndpoint(name=f"succ-ep-{self._name}",
                                                      remote_addr=message.peer_tuple[1], acceptor=False,
                                                      multithreading=True,
                                                      buffer_size=10000)
            self._successor_endpoint.start()
        self._find_succ(*message.peer_tuple, message.message_id)

    def _process_notify(self, message: Chordmessage):
        """Processes the received node from a notify message. If the node is not self but could be
        :param message:
        :return:
        """
        if message.peer_tuple[0] != self._id:
            self._stabilize(*message.peer_tuple)
            # send stabilize with received peer. received peer is predecessor of other peer

    def _process_stabilize(self, message: Chordmessage):
        """Manages incoming stabilize messages. Updates the predecessor of a peer or initiates a new notify message.

        :param message: Contains the id of a peer who believes self is its successor.
        """
        # if pred is still self change!!
        if self._predecessor == (self._id, self._addr):
            self._predecessor = message.peer_tuple
        # if (peer id is my pred (I am succ) or between me and my pred) update my pred
        elif self._is_succ(message.peer_tuple[0]):
            self._predecessor = message.peer_tuple
            ep = self._get_ep_if_exists(*message.peer_tuple)
            if ep is None:
                self._predecessor_endpoint = StreamEndpoint(
                    name=f"{self._name}-predecessor-ep-{np.random.randint(0, 100)}",
                    remote_addr=self._addr, acceptor=False, multithreading=True,
                    buffer_size=10000)
            else:
                self._predecessor_endpoint = ep
        else:
            return

    def _process_find_succ_res(self, message: Chordmessage):
        """

        :param message:
        :return:
        """
        if message.message_id not in self._sent_messages:
            return
        msg_origin, _, msg_finger = self._sent_messages.pop(message.message_id, (None, None, None))
        if msg_origin == MessageOrigin.JOIN:
            self._successor = message.peer_tuple
            self._stabilize(*message.peer_tuple)
        if msg_origin == MessageOrigin.FIX_FINGERS:
            self._update_finger(message.peer_tuple, msg_finger)

    def _process_find_succ_req(self, message: Chordmessage):
        """

        :param message:
        :return:
        """
        # call find succ with peer from message as target
        self._find_succ(*message.peer_tuple, message.message_id)

    def run(self, bootstrap_addr: tuple[str, int] = None):
        """Starts the chord peer, joining an existing chord ring or bootstrapping one, before starting the periodic
        loop to maintain the ring and process incoming messages.

        :param bootstrap_addr: Remote address of peer of existing ring to jon to, otherwise None.
        """
        last_refresh_time = time()
        self._endpoint_server.start()
        if bootstrap_addr is not None:
            self._join(bootstrap_addr)
        while True:
            # check here for departure or not, if yes depart.

            curr_time = time()

            if curr_time - last_refresh_time >= 30 and self._successor is not None:
                self._fix_fingers()
                self._stabilize(*self._successor)
                self._notify(*self._predecessor)
                last_refresh_time = curr_time

            r_ready, _ = self._endpoint_server.poll_connections()
            # TODO what about other eps opened by self? eg finger-eps that recv? are they in ep-server?
            for addr in r_ready:
                try:
                    message = typing.cast(Chordmessage, r_ready[addr].receive())
                    msg_type = message.message_type
                    self._logger.info(f"Received Message of Type {msg_type} with peer {message.peer_tuple}")
                    self._logger.info(
                        f"PEER STATS::id::{self._id}, predecessor: {self._predecessor}, successor: {self._successor}")

                    match msg_type:
                        case MessageType.JOIN:
                            self._process_join(message)
                        case MessageType.FIND_SUCC_RES:
                            self._process_find_succ_res(message)
                        case MessageType.FIND_SUCC_REQ:
                            self._process_find_succ_req(message)
                        case MessageType.STABILIZE:
                            self._process_stabilize(message)
                        case MessageType.NOTIFY:
                            self._process_notify(message)
                except (TypeError, RuntimeError) as e:
                    self._logger.error(f"{e.__class__.__name__} ({e}) :: in run during receive")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    in_name = input('Enter peer name:')
    in_port = input('Enter peer port:')
    in_remote_port = input('Enter peer remote port:')
    in_addr = "127.0.0.1"
    # ("127.0.0.1", 32000)
    # ("127.0.0.1", 13000)

    peer = Chordpeer(name=in_name, addr=(in_addr, int(in_port)), max_fingers=16)
    if in_remote_port == '':
        peer.run()  # start as first chord peer
    else:
        peer.run((in_addr, int(in_remote_port)))  # join existing chord
