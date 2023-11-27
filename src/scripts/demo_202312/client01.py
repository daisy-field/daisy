import logging
from src.federated_ids_components import FederatedOnlineClient


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)
    # TODO
    client_01 = FederatedOnlineClient()