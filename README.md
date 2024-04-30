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

*Basically: You provide the model and data source, plus any other customizations
you want done following the defined interfaces, and you are set!*

There also is a toolbox of various implementations of all these interfaces, Execution 
of code is be done through pure python or wrapped inside one or multiple docker containers. 


## Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.

```shell
git clone https://github.com/daisy-field/daisy.git
pip install /path/to/daisy
```

```shell
pip install pip install /path/to/daisy[cuda]
```

Here you should say what actually happens when you execute the code above.

### Initial Configuration

Some projects require initial configuration (e.g. access tokens or keys, `npm i`).
This is the section where you would document those requirements.

## Developing

Here's a brief intro about what a developer must do in order to start developing
the project further:

```shell
git clone https://github.com/daisy-field/daisy.git
cd daisy
pip install -e .[dev]
```

And state what happens step-by-step.

### Building

If your project needs some additional steps for the developer to build the
project after some code changes, state them here:

```shell
docker build .
```

```shell
docker build . --build-arg BUILD_VERSION=gpu
```

Here again you should state what actually happens when the code above gets
executed.

### Deploying / Publishing

In case there's some step you have to take that publishes this project to a
server, this is the right time to state it.

```shell
packagemanager deploy awesome-project -s server.com -u username -p password
```

And again you'd need to tell what the previous code actually does.

## Features

What's all the bells and whistles this project can perform?
* What's the main functionality
* You can also do another thing
* If you get really randy, you can even do this

## Configuration

Here you should write what are all of the configurations a user can enter when
using the project.

#### Argument 1
Type: `String`  
Default: `'default value'`

State what an argument does and how you can use it. If needed, you can provide
an example below.

Example:
```bash
awesome-project "Some other value"  # Prints "You're nailing this readme!"
```

#### Argument 2
Type: `Number|Boolean`  
Default: 100

Copy-paste as many of these as you need.

## Contributing

When you publish something open source, one of the greatest motivations is that
anyone can just jump in and start contributing to your project.

These paragraphs are meant to welcome those kind souls to feel that they are
needed. You should state something like:

"If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome."

If there's anything else the developer needs to know (e.g. the code style
guide), you should link it here. If there's a lot of things to take into
consideration, it is common to separate this section to its own file called
`CONTRIBUTING.md` (or similar). If so, you should say that it exists here.

## Links

Even though this information can be found inside the project on machine-readable
format like in a .json file, it's good to include a summary of most useful
links to humans using your project. You can include links like:

- Project homepage: https://your.github.com/awesome-project/
- Repository: https://github.com/your/awesome-project/
- Issue tracker: https://github.com/your/awesome-project/issues
    - In case of sensitive bugs like security vulnerabilities, please contact
      my@email.com directly instead of using issue tracker. We value your effort
      to improve the security and privacy of this project!
- Related projects:
    - Your other project: https://github.com/your/other-project/
    - Someone else's project: https://github.com/someones/awesome-project/


## Licensing

One really important part: Give your project a proper license. Here you should
state what the license is and how to find the text version of the license.
Something like:

"The code in this project is licensed under MIT license."