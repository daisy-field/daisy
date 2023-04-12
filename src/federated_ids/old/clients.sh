#!/bin/bash

python Data_simulator.py 2 64434 & 
python Data_simulator.py 5 64436 & 

python Client.py 2 64433 64434 & 
python Client.py 5 64435 64436  

sleep 20