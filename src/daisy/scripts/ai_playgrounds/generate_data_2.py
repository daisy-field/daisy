from daisy.data_sources import DataSource
from daisy.data_sources import PcapHandler, march23_events
from daisy.data_sources import pyshark_map_fn
from daisy.data_sources import SimpleDataProcessor, CohdaProcessor
from daisy.data_sources import CSVFileRelay
from time import sleep
import logging
from datetime import datetime

import pandas as pd

def cohda_reduce(d_point: dict, events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], int]], client_id: int):
    for event in events:
        client, (start_time, end_time), protocols, addresses, label = event
        if (
                client == client_id
                and start_time
                <= datetime.strptime(d_point["meta.time"], "%Y-%m-%d %H:%M:%S.%f")
                <= end_time
                and any([x in d_point["meta.protocols"] for x in protocols])
                and all([x in d_point["ip.addr"] for x in addresses])
        ):
            d_point["label"] = label
            break
    return d_point

def cohda_filter(d_point: dict):
    d_point["label"] = 0
    return d_point


logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


client_id = 2
pro = "simple"
handler = PcapHandler(f"/home/daisy_datasets/v2x_2023-03-06/diginet-cohda-box-dsrc{client_id}")

processor = None
if pro != "simple":
    processor = SimpleDataProcessor(pyshark_map_fn(), lambda d_point: cohda_filter(d_point), lambda d_point: cohda_reduce(d_point, events=march23_events, client_id=client_id))
elif pro != "cohda":
    processor = CohdaProcessor(client_id=client_id, events=march23_events)
else:
    print("Error")

csv_file_name = f"csv-example-full-{pro}-dsrc_{client_id}.csv"

source = DataSource(source_handler=handler, data_processor=processor)
csv = CSVFileRelay(data_source=source,
                   target_file=csv_file_name,  # Output Datei
                   overwrite_file=True,  # Bool, ob der Output überschrieben werden soll (praktisch zum testen)
                   separator=",",  # der separator in der csv
                   default_missing_value="") # Welcher Value wird verwendet, wenn die Datenpunkt keinen Wert in der Spalte hat

csv.start()
csv.join()
csv.stop()

df = pd.read_csv(csv_file_name)

labels = df['label'].unique()

print(labels)
print("Test")
