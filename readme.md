

# V2X-Vehicle (TL-Communication)

Simple ROS-less Python service to make traffic light states transmitted by the local cohda box to the service available
to other clients.

## Installation and Setup

Note: If one wants to run this tool with actual live-data, the service's pc has to be connected to a cohda box over the
network and the cohda box has to be configured in a way that it forwards all DSRC packets to the server as UDP packets.
See the DSRCpy package documentation for more information regarding that.

### Docker

#### Quick Setup

1. Clone this project.
2. Log into the docker registry of ``registry.gitlab.dai-labor.de``.
3. Pull the latest images and deploy it directly via ``./deploy``.

#### Manual Setup

Allows development directly without first building the images over the deployment pipeline before pulling it. It works,
but why not just go for a pure python setup instead?

1. Clone this project, including the submodules.
2. Build and start everything:
    - ``docker compose -f docker-compose.dev.yml up`` development container with preloaded DSRC packets (outdated)
    - ``docker compose -f docker-compose.yml up`` production container that listens to the cohda box for messages.

### Python

Runs without issues under python3 (3.8.10) and pip3 (20.0.2)

1. Clone this project and set up a virtual environment with the given requirements. Make sure to pull the submodules as
   well.
2. Install the [DSCRpy package](https://gitlab.dai-labor.de/diginet-ps/dsrcpy) from the submodule via ``pip``.
3. Run ``trafficlight_service.py --help`` for a basic info into the usage (args).

## Usage

Communication with the service is done via UDP and using a simple JSON request/reply format:

**Sample Request:**
The ID of the TL ID requested is from the provided HD-map. Note that if the map changes, the mapping has to be updated (
see `src/trafficlight_service/mapping.json`).

```
{"tl_id": "10000000000000063"}
```

**Sample Reply:**
The reply is synchronous (UDP is simply used for ease of use) and features a list of all known phases and their start
and end time stamp. Note that start times in this sample are wrong due to being from the demo.

```
[{"color": "green", "startTime": 1663924629.2965586, "endTime": 1659953487.0}, {"color": "yellow", "startTime": 1663924629.2966504, "endTime": 1659953596.0}]
```