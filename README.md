# DAISY

> A Framework for Fully **D**istributed, **A**nomaly-Based **I**ntrusion Detection in
> **S**ecurit**y**-Oriented Edge Computing Environments.

[![CI](https://github.com/daisy-field/daisy/actions/workflows/ci.yml/badge.svg)](https://github.com/daisy-field/daisy/actions/workflows/ci.yml)
[![Coverage Status](.github/workflows/coverage-badge.svg)](.github/workflows/coverage-badge.svg)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://github.com/daisy-field/daisy/blob/main/LICENSE.txt)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

DAISY aims to be an end-to-end framework to design, develop, and execute distributed
intrusion detection systems (IDS) of varying topologies, in an edge-optimized fashion.
All of which is done in python and done generically.

*Basically: You provide the model along some data sources, plus any other customizations
you want done following the defined interfaces, and you are set!*

For the latter, there is a large toolbox of various (example) implementations for all
these interfaces. Execution i.e. rollout is done through pure python or wrapped inside
one or multiple docker containers.

> [!CAUTION]
> DAISY is currently in the pre-alpha stage, as lots of features are still under heavy 
> testing. Many features more are also still located in their own feature 
> branches or 
> located in private mirrors. Use it at your own risk, [feedback](#contributing) 
> is always welcome. Stay tuned for more cool stuff in the future!

# Table of Contents

1. [Installing / Getting Started](#installing--getting-started)
2. [Developing](#developing)
    1. [Building](#building)
3. [Frequently Asked Questions](#frequently-asked-questions)
4. [Configuration](#configuration)
    1. [Minimum Working Example](#minimum-working-example)
5. [Contributing](#contributing)
6. [Licensing](#licensing)

## Installing / Getting Started

DAISY supports `pip install` under
[Python 3.12](https://www.python.org/downloads/release/python-3120/) and it can be
installed in the way below. DAISY is also supported through a Docker container and
the project can be used out of the box after [building it](#building). Generally, it is
recommended to use a
[virtual environment](https://docs.python.org/3.12/library/venv.html) for any python
project. For CUDA-enabled GPU cards (mainly on Ubuntu and various Linux distributions),
there is additional support directly integrated into DAISY via the `[cuda]` option;
this functionality requires NVIDIA® GPU drivers and is supported through and by
[Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/README.md). For live 
traffic capture and pcap reading functionality, 
[Tshark](https://tshark.dev/setup/install/) is required.

```shell
git clone https://github.com/daisy-field/daisy.git

# python 3.12 setup
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12

# venv setup (recommended)
sudo apt install python3.12-venv
python3.12 -m venv venv
source venv/bin/activate

pip install /path/to/daisy
pip install /path/to/daisy[cuda]  # gpu support (optional)
```

Afterward, the demo scripts are added to the shell path and may be executed, such as:

```shell
demo_202303_client -h
model_aggr_server -h
```

Follow the instructions to perform an initial demo. There is also a [minimum working
example](#minimum-working-example) with all necessary components for a setup of two
federated detection nodes, the aggregation servers, and a dashboard to display the
results. Note that some of the demos require additional input as in data sources
which are not part of this project, but require a path to a local version of our 
[data set repository](https://github.com/daisy-field/datasets).

## Developing

Since DAISY supports regular `pip`, to develop the project further or to adapt the
example scripts directly, one should install it in edit mode
[(-e flag)](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e). Again, it is
once again recommended to either use a
[virtual environment](https://docs.python.org/3.12/library/venv.html) or any of the
alternatives, especially when developing.

```shell
git clone https://github.com/daisy-field/daisy.git
cd daisy

# python 3.12 setup
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12

# venv setup (recommended)
sudo apt install python3.12-venv
python3.12 -m venv venv
source venv/bin/activate

pip install -e .[dev]
```

This will add any external and internal dependencies to the `PATH`, besides installing
development tools such as style checker, unit tests, and more. If you want to
automatically let your commit be checked before pushing changes upstream via the
project's githooks, one additional command must be executed after installing DAISY:

```shell
pre-commit install
```

Since these checks performed by `pre-commit` will also be repeated upstream, this step
is highly recommended to avoid any failed checks during pull requests.

### Building

Since DAISY can be installed via `pip -e`, any code changes are immediately available.
However, if you want to use DAISY in docker, some of the image's layers must be rebuilt:

```shell
docker build -t daisy .
docker build -t daisygpu . --build-arg build_version=gpu  # gpu support (optional)
```

Afterward, the docker container can be run with in interactive shell mode to be used
like after installing DAISY from the shell (see above):

```shell
docker run -it --network host --name daisy daisy bash
```

You can also deploy one of the bundled demo setups, including the [minimum working
example](#minimum-working-example), directly via `docker compose` from the 
compose files provided in the demos directory. This forces a default configuration 
of all the components, except of the path to the data sets that need to be passed via 
an environment variable:

```shell
# Arg Setup
export DATASET_PATH=/path/to/datasets/v2x_2023-03-06
export BUILD_VERSION=gpu  # gpu support (optional)

docker compose -f demos/v2x_2023-03-06.yml up
```

Once fully deployed and running, the dashboard will be reachable over 
http://localhost:8000/.

[//]: # ()

[//]: # (### Deploying / Publishing)

[//]: # ()

[//]: # (In case there's some step you have to take that publishes this project to a)

[//]: # (server, this is the right time to state it.)

[//]: # ()

[//]: # (```shell)

[//]: # (packagemanager deploy awesome-project -s server.com -u username -p password)

[//]: # (```)

[//]: # ()

[//]: # (And again you'd need to tell what the previous code actually does.)


[//]: # (## Features)

[//]: # ()

[//]: # (What's all the bells and whistles this project can perform?)

[//]: # (* What's the main functionality)

[//]: # (* You can also do another thing)

[//]: # (* If you get really randy, you can even do this)

[//]: # ()

## Configuration

### Minimum Working Example

```shell
dashboard

pred_aggr_server

model_aggr_server

eval_aggr_server

demo_202303_client --clientId 5 --pcapBasePath /path/to/datasets/v2x_2023-03-06 \
--modelAggrServ localhost --updateInterval 5 --evalServ localhost --aggrServ localhost

demo_202303_client --clientId 2 --pcapBasePath /path/to/datasets/v2x_2023-03-06 \ 
--modelAggrServ localhost --updateInterval 5  --evalServ localhost --aggrServ localhost
```

## Frequently Asked Questions

#### 1. Dashboard not starting (e.g. crossref error)

Try to use 127.0.0.1 instead of localhost in address. Restart dashboard, try to use
different browser (Chromium based browsers are recommended). Deactivate ad blockers and
enable JavaScript.

#### 2. Module 'ml_dtypes' has no attribute 'bfloat16' when starting dashboard

Check installation of tensorflow (version & correct installation in venv)

#### 3. Socket Trying to (re-)establish connection

Somehow sockets cannot make a connection to other components. Common windows problem (we
recommend to use WSL).
Check settings of protected folder access, try to restart components/computer.

#### 4. PCAP files aren't read

For network traffic, PyShark is used. This is a library using tshark in the background.
This means that it is dependent on the tshark installation.
It may be required to execute the code with root/admin permissions, as tshark might be
configured to deny non-root users to use its features.
On Windows machines it was observed, that pyshark has trouble using tshark, despite
correct installation and path variables. WSL or Linux might be required in these cases.
Note that CSV files do not use PyShark and should work regardless of the environment/OS
used. CSVs are, therefore, generally recommended over PCAP.

#### 5. Live Network traffic isn't captured

Refer to question 4, as the Live Network capture uses PyShark and suffers from the same
problems.

#### 6. Script X isn't producing any data

Depending on the script, this can have different reasons. Generally, if it either uses
PCAP files as the input or uses live network traffic, it probably uses PyShark. Refer to
question 4 for this.

## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests and bug reports are warmly welcome.

Note that DAISY uses the [
*Black*](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
code style through the [Ruff](https://docs.astral.sh/ruff/) formatter.


[//]: # ()

[//]: # (## Links)

[//]: # ()

[//]: # (Even though this information can be found inside the project on machine-readable)

[//]: # (format like in a .json file, it's good to include a summary of most useful)

[//]: # (links to humans using your project. You can include links like:)

[//]: # ()

[//]: # (- Project homepage: https://your.github.com/awesome-project/)

[//]: # (- Repository: https://github.com/your/awesome-project/)

[//]: # (- Issue tracker: https://github.com/your/awesome-project/issues)

[//]: # (    - In case of sensitive bugs like security vulnerabilities, please contact)

[//]: # (      my@email.com directly instead of using issue tracker. We value your effort)

[//]: # (      to improve the security and privacy of this project!)

[//]: # (- Related projects:)

[//]: # (    - Your other project: https://github.com/your/other-project/)

[//]: # (    - Someone else's project: https://github.com/someones/awesome-project/)

[//]: # ()

## Licensing

The code in this project is licensed under the Mozilla Public License
Version 2.0 [(MPL 2.0)](https://github.com/daisy-field/daisy/blob/main/LICENSE.txt)