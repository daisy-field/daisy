import logging
import threading
from pathlib import Path

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataSource


class DataSourceRelay:  # TODO comment and variable declarations
    """A union of a data source and a stream endpoint to retrieve data points from the former and relay them over the
    latter. This allows the disaggregation of the actual datasource from the other processing steps. For example, the
    relay could be deployed with our without an actual processor on another host and the data is forwarded over the
    network to another host running a data source with a SimpleRemoteSourceHandler to receive and further process the
    data. This chain could also be continued beyond a single host pair.
    """
    _logger: logging.Logger

    _data_source: DataSource
    _endpoint: StreamEndpoint

    _relay: threading.Thread
    _started: bool

    def __init__(self, data_source: DataSource, endpoint: StreamEndpoint, name: str = ""):
        """Creates a new data source relay. If either isn't provided, one can also provide the basic parameters for the
        creation of data source and/or endpoint.

        :param name: Name of data source relay for logging purposes.
        :param data_source: Data source to relay data points from.
        :param endpoint: Streaming endpoint to which data points are relayed to.
        :param source_handler: Actual source that provisions data points to data source. Fallback from data source.
        :param generator: Generator object from which data points are retrieved, fallback from source handler.
        :param data_processor: Processor containing the methods on how to process individual data points.
        :param process_fn: Processor functioning to process individual data points. If neither processor nor function
        provided, defaults to NOP.
        :param addr: Local address of new endpoint. Fallback from endpoint.
        :param remote_addr: Address of remote endpoint to be connected to. Fallback from endpoint.
        :param multithreading: Enables multithreading of data source and endpoint for speedup.
        :param buffer_size: Size of shared buffers in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing data source relay...")

        self._started = False

        self._data_source = data_source
        self._endpoint = endpoint

        self._logger.info("Data source relay initialized.")

    def start(self):
        """Starts the data source relay along any other objects in this union (data source, endpoint). Non-blocking, as
        the relay is started in the background to allow the stopping of it afterward.
        """
        self._logger.info("Starting data source relay...")
        if self._started:
            raise RuntimeError(f"Relay has already been started!")
        self._started = True
        try:
            self._data_source.open()
        except RuntimeError:
            pass
        try:
            self._endpoint.start()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("Data source relay started.")

    def stop(self):
        """Closes and stops the data source and the endpoint and joins the relay thread into the current thread. Can be
        restarted (and stopped) and arbitrary amount of times.
        """
        self._logger.info("Stopping data source relay...")
        if not self._started:
            raise RuntimeError(f"Endpoint has not been started!")
        self._started = False
        try:
            self._data_source.close()
        except RuntimeError:
            pass
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass

        self._relay.join()
        self._logger.info("Data source relay stopped.")

    def _create_relay(self):
        """Actual relay, directly forwards data points from its data source to its endpoint (both might be async).
        """
        self._logger.info("Starting to relay data points from data source...")
        for d_point in self._data_source:
            try:
                self._endpoint.send(d_point)
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Data source exhausted, or relay closing...")

    def __del__(self):
        if self._started:
            self.stop()


class FileRelay:  # TODO add comments and variable declarations
    _logger: logging.Logger

    _data_source: DataSource
    _file: Path

    _relay: threading.Thread
    _started: bool

    def __init__(self, target_file: str, data_source: DataSource, name: str = "", overwrite_file: bool = False):
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing file relay...")

        self._started = False

        if target_file is None or not target_file:
            raise ValueError("File to write to required.")
        self._file = Path(target_file)
        if self._file.is_dir():
            raise ValueError("File path points to a directory instead of a file.")
        parent_dir = Path(*self._file.parts[:-1])
        parent_dir.mkdir(parents=True, exist_ok=True)  # create parent directories
        try:  # Create the file. If it exists, FileExistsError will be raised. If it is invalid, FileNotFoundError will be raised
            self._file.touch(exist_ok=False)
        except FileNotFoundError:
            raise ValueError("File points at an invalid path.")
        except FileExistsError:
            if not overwrite_file:
                raise ValueError("File already exists and should not be overwritten.")

        self._data_source = data_source

        self._logger.info("File relay initialized.")

    def start(self):
        self._logger.info("Starting file relay...")
        if self._started:
            raise RuntimeError(f"Relay has already been started!")
        self._started = True
        try:
            self._data_source.open()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("File relay started.")

    def stop(self):
        self._logger.info("Stopping file relay...")
        if not self._started:
            raise RuntimeError(f"Relay has not been started!")
        self._started = False
        try:
            self._data_source.close()
        except RuntimeError:
            pass

        self._relay.join()
        self._logger.info("File relay stopped.")

    def _create_relay(self):
        self._logger.info("Starting to relay data points from data source...")
        # with self._file.open("b") as file:
        for d_point in self._data_source:
            try:
                self._logger.info(d_point)
                # TODO need to get the headers for the CSV here and write each new line to file
               # self._logger.info(type(d_point))
               # self._logger.info(dir(d_point))
               # self._logger.info(getattr(d_point, "eth"))
               # self._logger.info(dir(getattr(d_point, "eth")))
               # try:
               #     self._logger.info(getattr(getattr(d_point, "eth"), "addra"))
               # except AttributeError:
               #     self._logger.info("Unknown thingy")
                # file.write(d_point)  # TODO make this to bytes
                # TODO write to file
                pass
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Data source exhausted, or relay closing...")

    def __del__(self):
        if self._started:
            self.stop()
