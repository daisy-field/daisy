from time import time
from typing import Optional
from uuid import uuid4

from communication import StreamEndpoint, EndpointServer

from daisy.chord.chord_peer import MessageOrigin


def check_if_peer_is_predecessor(pred_id: int, peer_id: int, new_pred_id: int) -> bool:
    """Checks whether a peer is the precessor of self.

    Note: If the predecessor of self has not been changed since init True will be returned.

    :param peer_id:
    :param pred_id:
    :param new_pred_id: Id of peer check successor relation
    :return: True if self is successor of peer, else False
    """
    if pred_id == peer_id and new_pred_id != peer_id:
        return True

    return (
        (peer_id < pred_id) and (new_pred_id not in range(peer_id + 1, pred_id + 1))
    ) or (new_pred_id in range(pred_id + 1, peer_id + 1))


def check_if_peer_is_successor(succ_id: int, peer_id: int, new_succ_id: int) -> bool:
    """Checks whether a node is the successor of self.
    Note: If the successor of self is None True will be returned.
    :param succ_id:
    :param new_succ_id:
    :param peer_id: Id of peer check predecessor relation
    :return: True if self is predecessor of peer, else false.
    """
    if succ_id is None and new_succ_id != peer_id:  # if no pred set
        return True
    return (
        (peer_id > succ_id) and (new_succ_id not in range(succ_id + 1, peer_id + 1))
    ) or (new_succ_id in range(peer_id + 1, succ_id + 1))


def check_if_peer_is_between(peer_id: int, intermediate_id: int, boundary_id) -> bool:
    return (
        peer_id > boundary_id and intermediate_id in range(boundary_id - 1, peer_id)
    ) or (intermediate_id not in range(peer_id, boundary_id - 1))


def fingertable_to_string(
    fingertable: dict[int, tuple[int, tuple[str, int], StreamEndpoint]],
) -> list[str]:
    return [("(" + f"{key}: {fingertable[key][0]}" + ")") for key in fingertable.keys()]


def close_and_remove_endpoint(
    ep_server: EndpointServer, ep_addr: tuple[str, int], ep: StreamEndpoint
):
    """
    :param ep_server:
    :param ep_addr: Address of endpoint to remove (used for lookup in EndpointServer)
    :param ep: StreamEndpoint to remove
    """
    ep_in_epserver = ep_server.get_connections([ep_addr]).get(ep_addr)
    if ep_in_epserver is None:
        if ep.poll()[0][0] and ep is not None:
            ep.stop(shutdown=True)


def cleanup_dead_messages(
    sent_messages: dict[uuid4, tuple[MessageOrigin, time, Optional[int]]], ttl: float
):
    """Checks whether sent messages can be forgotten and deletes them."""
    for key in list(sent_messages):
        if sent_messages.get(key, None) is not None:
            if time() - sent_messages[key][1] > ttl:
                sent_messages.pop(key)


def get_readable_endpoints(
    finger_values,
    succ_ep: StreamEndpoint,
    pred_ep: StreamEndpoint,
    ep_server: EndpointServer,
) -> set[StreamEndpoint]:
    """Receives on all available endpoints where there is something to receive. May return an empty list if no messages were received.

    :return: List of StreamEndpoints with readable buffer
    """
    readable_endpoints = {
        finger_ep for _, _, finger_ep in finger_values if finger_ep.poll()[0][1]
    }

    if (succ_ep is not None) and succ_ep.poll()[0][1]:
        readable_endpoints.add(succ_ep)
    if (pred_ep is not None) and pred_ep.poll()[0][1]:
        readable_endpoints.add(pred_ep)
    r_ready, _ = ep_server.poll_connections()
    for addr in r_ready:
        readable_endpoints.add(r_ready[addr])
    return readable_endpoints
