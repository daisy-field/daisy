import json
import pyshark
from pyshark.packet.packet import Packet

# def render_csv_row(p: Packet) -> str:
#     return f"{p.captured_length}," \
#            f"{p.show}," \
#            f"{}," \
#            f"{}," \
#            f"{}," \
#            f"{}," \
#            f"{},"

def pyshark_capture():
    capture = pyshark.LiveCapture(interface='any', use_json=True)
    p: Packet
    for p in capture.sniff_continuously():
        print(p)

if __name__ == '__main__':
    pyshark_capture()
