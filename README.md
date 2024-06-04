# DAISY

> A Framework for Fully **D**istributed, **A**nomaly-Based **I**ntrusion Detection in
> **S**ecurit**y**-Oriented Edge Computing Environments.

[![CI](https://github.com/daisy-field/daisy/actions/workflows/ci.yml/badge.svg)](https://github.com/daisy-field/daisy/actions/workflows/ci.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://github.com/daisy-field/daisy/blob/main/LICENSE.txt)
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


## Installing / Getting started

DAISY supports `pip install` **under Python 3.11** and can be set up the following way.
Note, generally it is recommended to use a virtual environment for any python project.

```shell
git clone https://github.com/daisy-field/daisy.git

pip install /path/to/daisy
pip install pip install /path/to/daisy[cuda]  # gpu support
```

Afterward, the demo scripts are added to the shell path and may be executed, such as:

```shell
demo_202312_client -h
model_aggr_server -h
```

Follow the instructions to perform an initial demo. Note that some of the demos require
additional input as in data sources which are not part of this project. DAISY is also
supported in docker and the project can be used out of the box after
[building it](#building).


## Developing

Since DAISY supports regular `pip`, to develop the project further, one should install
it in edit mode [(-e flag)](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e):

```shell
git clone https://github.com/daisy-field/daisy.git
cd daisy
pip install -e .[dev]
```

This will add any external and internal dependencies to the `PATH`, besides installing
development tools such as style checker, unit tests, and more. If you want to check your
commit before pushing changes upstream via the project's githooks, one additional
command must be executed after installing DAISY:

```shell
pre-commit install
```


### Building

Since DAISY can be installed via `pip -e`, any code changes are immediately available.
However, if you want to use DAISY in docker, some of the image's layers must be rebuilt:

```shell
docker build .
docker build . --build-arg BUILD_VERSION=gpu
```

Afterward, the docker container can be run with in interactive shell mode to be used
like after installing DAISY from the shell (see above).


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
[//]: # (## Configuration)

[//]: # ()
[//]: # (Here you should write what are all of the configurations a user can enter when)

[//]: # (using the project.)

[//]: # ()
[//]: # (#### Argument 1)

[//]: # (Type: `String`  )

[//]: # (Default: `'default value'`)

[//]: # ()
[//]: # (State what an argument does and how you can use it. If needed, you can provide)

[//]: # (an example below.)

[//]: # ()
[//]: # (Example:)

[//]: # (```bash)

[//]: # (awesome-project "Some other value"  # Prints "You're nailing this readme!")

[//]: # (```)

[//]: # ()
[//]: # (#### Argument 2)

[//]: # (Type: `Number|Boolean`  )

[//]: # (Default: 100)

[//]: # ()
[//]: # (Copy-paste as many of these as you need.)


## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

Note that DAISY uses the [*Black*](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code style through the [Ruff](https://docs.astral.sh/ruff/) formatter.


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


## Licensing

The code in this project is licensed under the Mozilla Public License
Version 2.0 [(MPL 2.0)](https://github.com/daisy-field/daisy/blob/main/LICENSE.txt)