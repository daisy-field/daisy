import random
from time import sleep

import requests

url = 'http://127.0.0.1:8000/accuracy/'
for i in range (0,10000):
    val = random.uniform(0, 1)

    myobj = {'accuracy': str(val)}
    x = requests.post(url, data = myobj)
    sleep(1)