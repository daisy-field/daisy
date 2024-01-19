# syntax=docker/dockerfile:1

FROM python:3.10

WORKDIR /app
COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install dsrcpy/

EXPOSE 8800/udp
EXPOSE 4400/udp

CMD [ "python3", "src/trafficlight_service/trafficlight_service.py", "--mappingPath", "src/trafficlight_service/mapping.json", "--debug", "True"]