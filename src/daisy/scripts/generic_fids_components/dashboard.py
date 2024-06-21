# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Simple wrapper around the django command-line utility to launch the django-based
dashboard, preparing the necessary files and finally running the server.

The dashboard itself is then available on port 8000.

Author: Seraphin Zunzer, Fabian Hofmann
Modified: 18.06.24
"""

import os


def create_dashboard():
    """Creates and runs the django server (on port 8000), auto-generating any
    necessary changes to the database.

    Since django has an obnoxious need to maintain its own namespace, does not call
    the django utility functions directly, but resorts to executing the actual files
    to decouple them as much as possible from any possible interference.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dashboard_path = os.path.join(
        dir_path, "../../federated_ids_components/dashboard/manage.py"
    )

    os.system(f"python {dashboard_path} makemigrations")
    os.system(f"python {dashboard_path} migrate")
    os.system(f"python {dashboard_path} runserver")


if __name__ == "__main__":
    create_dashboard()
