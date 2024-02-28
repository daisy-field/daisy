# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import random
from time import sleep

import requests

url = 'http://127.0.0.1:8000/accuracy/'
k=0
for i in range (0,10000):
    val = random.uniform(0, 1)

    myobj = {'accuracy': str(k)}
    x = requests.post(url, data = myobj)
    sleep(1)
    k+=0.01