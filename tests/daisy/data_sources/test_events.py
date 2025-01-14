# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
from pyparsing import ParseException

from daisy.data_sources import EventHandler


@pytest.fixture
def event_parser():
    return EventHandler()._parser


class TestEventParser:
    @pytest.mark.parametrize("expression,error_expected", [("a=b", False), ("a", True)])
    def test_parser_syntax(self, event_parser, expression, error_expected):
        if error_expected:
            with pytest.raises(ParseException):
                event_parser.parse(expression)
        else:
            event_parser.parse(expression)


# =
# in
# and
# or
# not

# base word
# breacket word
# comparator
# boperator
# uoperator
# [bracket word]
# operand
# pars ( parser )
# pars + boperator + pars
# uoperator + pars

# synstax tests
# logic tests
# is the whole expression read?
