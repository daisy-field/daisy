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

        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True, c_timeout=30)
        self._max_fingers = max_fingers

        self._sent_messages = {}

        self._logger = logging.getLogger(name + "-Peer")

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
        if remote_addr == self._addr:
            self._logger.warning(
                f"Trying to send Message of Type {message.message_type} With {message.peer_tuple} to self. Aborting.")
            return
        try:
            # if not self._clean_up_dead_ep(ep):
            ep.send(message)
            self._logger.info(
                f"Sent Message of Type {message.message_type} From {self._name, self._id} To {remote_addr} With {message.peer_tuple}")
        except (AttributeError, RuntimeError) as e:  # attributeError if ep is none, runtimeError if ep is not started
            self._logger.warning(f"{e.__class__.__name__} ({e}) :: in _send_message: creating new ep to send.")

            ep_name = f"one-time-ep-{np.random.randint(0, 100)}"
            endpoint = StreamEndpoint(name=ep_name,
                                      remote_addr=remote_addr, acceptor=False, multithreading=False,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(message)
            self._logger.info(
                f"Sent Message of Type {message.message_type} From {self._name, self._id} To {remote_addr} With {message.peer_tuple}")
            threading.Thread(target=lambda: close_tmp_ep(endpoint, 10, 10), daemon=True).start()

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

    def cleanup_dead_messages(self):
        """Checks whether sent messages can be forgotten and deletes them.
        """
        for key in self._sent_messages.values():
            if self._sent_messages.get(key, None) is not None:
                if time() - self._sent_messages[key][1] > 20:
                    self._logger.info(f"Deleting Message: {self._sent_messages[key]}")
                    self._sent_messages.pop(key)

    def _notify(self, remote_id: int, remote_addr: tuple[str, int]):
        """Creates and sends a notify message to validate a peers predecessor. Used to repair successor pointers on the chord ring.
        """
        if self._predecessor[0] == self._id:
            self._logger.warning(f"Notify to self")
            return
        notify_message = Chordmessage(message_id=uuid4(), message_type=MessageType.NOTIFY,
                                      peer_tuple=self._predecessor)
        ep = self._get_ep(remote_id, remote_addr)
        self._send_message(ep, remote_addr, notify_message)

    def _stabilize(self, remote_id: int, remote_addr: tuple[str, int]):
        """Creates and sends a stabilize message to the sucessor of a peer, initiating reparation of its successor and respective predecessor pointers.

        TODO Succ may have failed in the meantime, ask next fingertable entry or predecessor
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
        ep = self._get_ep(remote_id, remote_addr)
        self._send_message(ep, remote_addr, stabilize_message)

    def _update_finger(self, finger_tuple: tuple[int, tuple[str, int]], finger_index: int):
        """Updates an entry in a peers fingertable by maintaining its StreamEndpoint and assigning current finger-values or removing the finger entirely.

        :param finger_tuple:
        :param finger_index:
        :return:
        """
        finger = self._fingertable.pop(finger_index, None)
        if finger is not None:
            finger_ids_in_ft = {finger_id for finger_id, finger_addr, finger_ep in self._fingertable.values()
                                if finger[2] is finger_ep}
            if len(finger_ids_in_ft) == 0:
                if finger[2].poll()[0][0]:
                    finger[2].stop(shutdown=True)

        try:
            ep = self._get_ep(*finger_tuple)
            self._fingertable[finger_index] = (finger_tuple[0], finger_tuple[1], ep)
        except TypeError as e:
            self._logger.error(f"{e.__class__.__name__} ({e}) :: in _update_finger")

    def _get_ep(self, remote_id: int, remote_addr: tuple[str, int]) -> Optional[StreamEndpoint]:
        """Checks whether a StreamEndpoint to a given peer exists and returns it. Returns fresh StreamEndpoint if none
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

        ep = {finger_ep for finger_id, finger_addr, finger_ep in self._fingertable.values()
              if remote_id == finger_id}
        try:
            return ep.pop()
        except KeyError as e:
            self._logger.error(f"{e.__class__.__name__} ({e}) :: in get_ep_if-exists: No ep found. Creating new ep")

            ep = StreamEndpoint(name=f"get-ep-{random.randint(0, 100)}",
                                remote_addr=remote_addr, acceptor=False, multithreading=False,
                                buffer_size=10000)
            ep.start()
            return ep

    def _clean_up_dead_ep(self, ep: StreamEndpoint) -> bool:  # currently not in useage
        """Searches through all available connections of a peer to see where a given StreamEndpoint is used and
        initiates cleanup of the StreamEndpoint if it has died.
        FIXME how do i actually know if ep is dropout? send ping?
        TODO predecessor und successor updates auslagern
        Note: may result in removal of successor or predecessor endpoint of a peer or empty fingertables.

        :param ep: StreamEndpoint suspected as dropout
        """
        if ep is None:  # no ep
            return True

        states, addrs = ep.poll()  # supposed to filter eps that are not dropouts, everything after this line assumes dropout ep
        if states[0] or states[1] or states[2]:  # may not work like this, ep can be not dropout but unavailable
            return False
        # everything below assumes dropout ep

        # clean through finger table
        dropout_keys = [finger_key for finger_key, finger in self._fingertable.items()
                        if ep is finger[2]]
        for do_key in dropout_keys:
            do_id, do_addr, do_ep = self._fingertable.pop(do_key, (None, None, None))
            if do_ep is None:
                continue
            self._close_and_remove_ep(do_addr, do_ep)  # close finger_eps that are ep

        # if ep is self._predecessor_endpoint:  # clean up predecessor ep if necessary
        #    self._close_and_remove_ep(self._predecessor[1], ep)
        #     self._predecessor = (self._id, self._addr)  # pred to self ist richtig, siehe process_stabilize
        #    self._predecessor_endpoint = None
        # if ep is self._successor_endpoint:  # clean up succ ep if necessary
        #    self._close_and_remove_ep(self._successor[1], ep)
        #    for finger in range(
        #        self._max_fingers):  # set some arbitrary finger as succ, will be repaired by stabilize/notify calls
        #        f_id, f_addr, f_ep = self._fingertable.get(finger, (None, None, None))
        #        if not self._clean_up_dead_ep(f_ep):
        #            self._successor = (f_id, f_addr)
        #            self._successor_endpoint = f_ep
        #            break
        #    if self._successor is None:
        #        self._successor = self._predecessor
        #        self._successor_endpoint = self._predecessor_endpoint
        return True

    def _close_and_remove_ep(self, ep_addr: tuple[str, int], ep: StreamEndpoint):
        """
        :param ep_addr: Address of StreamEndpoint to remove (used for lookup in EndpointServer)
        :param ep: StreamEndpoint to remove
        """
        ep_in_epserver = self._endpoint_server.get_connections([ep_addr]).get(ep_addr)
        if ep_in_epserver is None:
            if ep.poll()[0][0] and ep is not None:
                ep.stop(shutdown=True)
        # else:
        #   self._endpoint_server.close_connections([ep_addr])

    def _fix_fingers(self):
        """Iterates the finger indices and initiates lookups for the respective peers; defaults to the successor.

        Note: this method does not actually update any finger table entries. It will return early if a peer has no successor.
        """
        # if self._successor is None:
        self._logger.info(f"Starting to update Fingers... ")
        # return

        for i in range(self._max_fingers):
            finger = self._id + 2 ** i % (2 ** self._max_fingers)
            if self._is_succ(finger):
                self._update_finger(self._successor, i)
            else:
                message_id = uuid4()
                self._sent_messages[message_id] = (MessageOrigin.FIX_FINGERS, time(), i)
                self._find_succ(finger, self._addr, message_id)
                # update_finger happens in process find_succ res -> MessageOrigin.FIX_FINGERS
                # and in run periodically

    def _is_pred(self, peer_id: int) -> bool:
        """Checks whether a peer is the precessor of self.

        Note: If the predecessor of self has not been changed since init True will be returned.

        :param peer_id: Id of peer check successor relation
        :return: True if self is successor of peer, else False
        """
        if self._predecessor[0] == self._id and peer_id != self._id:
            self._logger.info(f"In _is_succ: self.predecessor has not yet been set; defaulting to true.")
            return True

        succ = ((self._id < self._predecessor[0]) and (
                peer_id not in range(self._id + 1, self._predecessor[0] + 1))) or (
                       peer_id in range(self._predecessor[0] + 1, self._id + 1))
        self._logger.info(f"In _is_succ: self is succ of {peer_id} is {succ}")
        return succ

    def _is_succ(self, peer_id: int) -> bool:
        """Checks whether a node is the successor of self.

        Note: If the successor of self is None True will be returned.

        :param peer_id: Id of peer check predecessor relation
        :return: True if self is predecessor of peer, else false.
        """
        if self._successor is None and peer_id != self._id:  # if no pred set
            self._logger.info(f"In _is_pred: self.predecessor has not yet been set; defaulting to true.")
            return True
        pred = ((self._id > self._successor[0]) and (peer_id not in range(self._successor[0] + 1, self._id + 1))) \
               or (peer_id in range(self._id + 1, self._successor[0] + 1))
        self._logger.info(f"In _is_pred: self is pred of {peer_id} is {pred}")
        return pred

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
                if self._node_is_between(finger[0], peer_id):
                    self._logger.info(f"In _get_closest_known_pred: returning finger {finger[0]}")
                    return finger
            except TypeError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in _get_closest_known_pred")
            except ValueError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in _get_closest_known_pred")
        self._logger.info(f"In _get_closest_known_pred: returning None")

    def _node_is_between(self, intermediate_peer_id, bounding_peer_id):
        # TODO better variable names
        return (self._id > bounding_peer_id and intermediate_peer_id in range(bounding_peer_id - 1, self._id)) or (
                intermediate_peer_id not in range(self._id, bounding_peer_id - 1))

    def _find_succ(self, peer_id: int, peer_addr: tuple[str, int], message_id: uuid4):
        """Checks whether a peer knows the sucessor of the given peer. If it knows the successor it will send back its (id, addr) pair, otherwise the next peer will be asked.

        Note: if a peer receives a find_succ request with its own id it will terminate the search.

        :param peer_addr: chord address of peer who initially sent the find_successor request
        :param peer_id: chord id of peer whose successor should be found
        :message_id:
        """
        if self._addr == peer_addr:  # addr not id, wenn id dann eigenen succ
            self._logger.info(f"In _find_succ: self._addr == peer_addr ")
            return
        # second condition: if two peers are in ring A should not send B to B as its successor
        elif peer_addr == self._predecessor[1] or self._is_pred(peer_id):
            self._logger.info(f"In _find_succ: found succ for {peer_id} as self ")
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          peer_tuple=(self._id, self._addr))
            ep = self._get_ep(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)

        elif self._is_succ(peer_id):
            self._logger.info(f"In _find_succ: found succ for {peer_id} as self._successor:{self._successor[0]} ")
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          peer_tuple=self._successor)
            ep = self._get_ep(peer_id, peer_addr)
            self._send_message(ep, peer_addr, succ_found_msg)  # not closing ep if it comes from check

        # idk, ask the closest pred of peer
        else:
            self._logger.info(f"In _find_succ: no succ found for {peer_id}. Trying to relay lookup.")
            closest_pred = self._get_closest_known_pred(peer_id)
            try:
                closest_pred[2].send(Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_REQ,
                                                  peer_tuple=(peer_id, peer_addr)))
            except AttributeError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in find_succ: failed to relay lookup; no ep for closest pred found in ft")
                succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                              peer_tuple=(self._id, self._addr))
                ep = self._get_ep(peer_id, peer_addr)
                self._send_message(ep, peer_addr, succ_found_msg)
            except RuntimeError as e:
                self._logger.error(
                    f"{e.__class__.__name__} ({e}) :: in find_succ: failed to relay lookup; endpoint not connected.")

    def _process_join(self, message: Chordmessage):
        """Handles incoming join messages. Sets the successor of self if it is None and initiates successor lookup for new node.

        :param message:
        :return:
        """

        if self._successor is None:
            self._logger.info(f"In _process_join: updating successor from None to {self._successor}.")
            self._successor = message.peer_tuple
            self._successor_endpoint = StreamEndpoint(name=f"succ-ep-{self._name}",
                                                      remote_addr=message.peer_tuple[1], acceptor=False,
                                                      multithreading=False,
                                                      buffer_size=10000)
            self._successor_endpoint.start()
        self._logger.info(f"In _process_join: initiating _find_succ for {message.peer_tuple}.")
        self._find_succ(*message.peer_tuple, message.message_id)

    def _process_notify(self, message: Chordmessage):
        """Handles incoming notify messages.

        :param message:
        :return:
        """
        if message.peer_tuple[0] == self._id:
            return
        if self._is_succ(message.peer_tuple[0]):
            self._logger.info(f"In _process_notify: updating successor from {self._successor} to {message.peer_tuple}.")
            self._successor = message.peer_tuple
            ep = self._get_ep(*message.peer_tuple)
            if ep is None:
                self._successor_endpoint = StreamEndpoint(
                    name=f"{self._name}-successor-ep-{np.random.randint(0, 100)}",
                    remote_addr=self._addr, acceptor=False, multithreading=False,
                    buffer_size=10000)
            else:
                self._successor_endpoint = ep

    def _process_stabilize(self, message: Chordmessage):
        """Handles incoming stabilize messages. Updates the predecessor of a peer or returns.

        :param message: Contains the id of a peer who believes the receiving peer is its successor.
        """
        # if pred is still self change!!
        if self._predecessor == (self._id, self._addr):
            self._predecessor = message.peer_tuple
            self._logger.info(f"In _process_stabilize: updated predecessor from self to {self._predecessor}.")
        # if (peer id is my pred (I am succ) or between me and my pred) update my pred
        elif self._is_pred(message.peer_tuple[0]) and (self._id != message.peer_tuple[0]):
            self._logger.info(
                f"In _process_stabilize: updating predecessor from {self._predecessor} to {message.peer_tuple}.")
            self._predecessor = message.peer_tuple
            ep = self._get_ep(*message.peer_tuple)
            if ep is None:
                self._predecessor_endpoint = StreamEndpoint(
                    name=f"{self._name}-predecessor-ep-{np.random.randint(0, 100)}",
                    remote_addr=self._addr, acceptor=False, multithreading=False,
                    buffer_size=10000)
            else:
                self._predecessor_endpoint = ep
        self._notify(*message.peer_tuple)

    def _process_find_succ_res(self, message: Chordmessage):
        """Handles incoming find successor response messages. Updates the successor of a peer and initiates stabilize, updates a finger in the fingertable, depending on the message origin.

        :param message: received Chordmessage
        :return:
        """
        if message.message_id not in self._sent_messages:
            self._logger.info(
                f"Recv dead Message {message.peer_tuple}, In _process_find_succ_res.")
            return

        msg_origin, _, msg_finger = self._sent_messages.pop(message.message_id, (None, None, None))

        self._logger.info(
            f"Recv {message.peer_tuple}, {msg_origin}, In _process_find_succ_res.")

        if msg_origin == MessageOrigin.JOIN:
            self._successor = message.peer_tuple
            self._successor_endpoint = StreamEndpoint(
                name=f"{self._name}-sucessor-ep-{np.random.randint(0, 100)}",
                remote_addr=self._addr, acceptor=False, multithreading=False,
                buffer_size=10000)
            self._logger.info(
                f"Updated successor to {self._successor}. In _process_find_succ_res.")
            self._stabilize(*message.peer_tuple)
        if msg_origin == MessageOrigin.FIX_FINGERS:
            self._logger.info(f"Initiating _update_finger for {msg_finger}. In _process_find_succ_res.")
            self._update_finger(message.peer_tuple, msg_finger)

    def run(self, bootstrap_addr: tuple[str, int] = None):
        """Starts the chord peer, joining an existing chord ring or bootstrapping a new one, before starting the periodic
        loop to maintain the ring and process incoming messages.

        :param bootstrap_addr: Remote address of peer to join existing ring, None to start new ring.
        """
        start = time()
        last_refresh_time = time()
        self._endpoint_server.start()

        if bootstrap_addr is not None:
            self._logger.info(f"Bootstrapaddr: {bootstrap_addr}")
            self._join(bootstrap_addr)

        while True:
            # check here for departure or not
            curr_time = time()
            if curr_time - last_refresh_time >= 30:
                self._fix_fingers()
                try:
                    self._stabilize(*self._successor)
                except TypeError as e:
                    self._logger.error(f"{e.__class__.__name__} ({e}) :: in _run")
                self.cleanup_dead_messages()
                last_refresh_time = curr_time
            sleep(1)
            received_messages = self._receive_on_all_endpoints(start)
            for message in received_messages:
                msg_type = message.message_type
                self._logger.info(f"In run: Recv {msg_type} with {message.peer_tuple}")
                self._logger.info(self.__str__())
                match msg_type:
                    case MessageType.JOIN:
                        self._process_join(message)
                    case MessageType.FIND_SUCC_RES:
                        self._process_find_succ_res(message)
                    case MessageType.FIND_SUCC_REQ:
                        self._process_find_succ_req(message)
                    case MessageType.STABILIZE:
                        self._process_stabilize(message)
                        self._logger.info("Post process stabilize: " + self.__str__())
                    case MessageType.NOTIFY:
                        self._process_notify(message)
                        self._logger.info("Post process notify: " + self.__str__())

    def _receive_on_all_endpoints(self, start: float):
        """Receives on all available endpoints where there is something to receive. May return an empty list if no messages were received.

        :param start: boot time of peer, for logging purposes.
        :return: List of Chordmessages for further processing
        """

        # TODO recv ohne Poll möglich?
        received_messages = []
        for _, _, finger_ep in self._fingertable.values():
            try:
                received_messages.append(typing.cast(Chordmessage, finger_ep.receive()))
            except (AttributeError, RuntimeError):
                pass
        try:
            received_messages.append(typing.cast(Chordmessage, self._successor_endpoint.receive()))
        except (AttributeError, RuntimeError):
            pass
        try:
            received_messages.append(typing.cast(Chordmessage, self._predecessor_endpoint.receive()))
        except (AttributeError, RuntimeError):
            pass
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            try:
                received_messages.append(typing.cast(Chordmessage, r_ready[addr].receive()))
            except RuntimeError:
                pass

        if len(received_messages) > 0:
            self._logger.info(
                f"Received {len(received_messages)} Messages")
        return received_messages

    def _process_find_succ_req(self, message: Chordmessage):
        """Handles incoming find successor request messages. Terminates the search if the request contains self, or initiates new find successor step.

        :param message: received Chordmessage
        """
        # call find succ with peer from message as target
        if message.peer_tuple[0] == self._id:
            self._logger.info(f"In _process_find_succ_req: terminating search because find_succ on self.")
            return
        self._logger.info(f"Init _find_succ for {message.peer_tuple[0]}, In _process_find_succ_req.")
        self._find_succ(*message.peer_tuple, message.message_id)

    def __str__(self):
        return (f"ChordPeer {self._name}:\n" +
                f"id: {self._id}, \n" +
                f"predecessor: {self._predecessor}, \n" +
                f"successor: {self._successor}, \n" +
                f"fingers: {[('(' + str(key) + ') ') for key in self._fingertable.keys()]}, \n" +
                f"currently waiting for responses to {len(self._sent_messages)} messages.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--peerName",
                        help="Peer name")
    parser.add_argument("--peerPort", type=int, default=None,
                        help="Port of peer")
    parser.add_argument("--remotePort", type=int, default=None,
                        help="Port of remote peer")
    args = parser.parse_args()

    peer_ip = "127.0.0.1"

    peer = Chordpeer(name=args.peerName, addr=(peer_ip, args.peerPort), max_fingers=16)
    if args.remotePort is None:
        peer.run()  # start as first chord peer
    else:
        peer.run((peer_ip, args.remotePort))  # join existing chord
