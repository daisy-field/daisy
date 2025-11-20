from daisy.data_sources import CSVFileDataSource, DataProcessor, CSVFileRelay, DataHandler

with open("features.txt", "r") as f:
    lines = f.readlines()
    lines = [line.removesuffix("\n") for line in lines]

source = CSVFileDataSource(files="all_4.csv")
processor = DataProcessor()
handler = DataHandler(data_source=source, data_processor=processor)
relay = CSVFileRelay(target_file="processed.csv", overwrite_file=True, data_handler=handler, header_buffer_size=1000000, separator=";")
relay.start(blocking=True)
relay.stop()