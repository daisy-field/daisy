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
    @pytest.mark.parametrize(
        "expression,error_expected",
        [
            # Comparators
            ("feature=value", False),
            ("value in feature", False),
            # Binary Operators
            ("feature = value and value in feature", False),
            ("feature = value or value in feature", False),
            # Unary Operators
            ("not feature = value", False),
            # Rule word: Valid bracket_word option
            ("[feature with spaces] = [value with spaces]", False),
            # Parentheses
            ("(feature = value)", False),
            ("(feature = value) and value in feature", False),
            ("feature = value and (value in feature)", False),
            ("(feature = value) and (value in feature)", False),
            ("(feature = value and value in feature)", False),
            ("((feature = value) and value in feature)", False),
            ("not (feature = value and value in feature)", False),
            ("not (feature = value and not value in feature)", False),
            # Binary Operators as Comparators
            ("feature and value", True),
            ("feature or value", True),
            # Unary Operator wrong order
            ("feature=value not", True),
            # Rule base_word: Invalid characters
            ("fe[ature = value", True),
            ("fe]ature = value", True),
            ("fe ature = value", True),
            ("fe!ature = value", True),
            ('fe"ature = value', True),
            ("fe'ature = value", True),
            ("fe<ature = value", True),  # can be confused with an operator
            ("fe=ature = value", True),  # can be confused with an operator
            ("fe>ature = value", True),  # can be confused with an operator
            ("fe\\ature = value", True),
            ("fe(ature = value", True),
            ("fe)ature = value", True),
            # Rule bracket_word: Invalid characters
            ("[fe[ature] = value", True),
            ("[fe]ature] = value", True),
            ("[fe!ature] = value", True),
            ('[fe"ature] = value', True),
            ("[fe'ature] = value", True),
            ("[fe<ature] = value", True),
            ("[fe=ature] = value", True),
            ("[fe>ature] = value", True),
            ("[fe\\ature] = value", True),
            ("[fe(ature] = value", True),
            ("[fe)ature] = value", True),
            # Rule operand: Missing components
            ("feature", True),
            ("feature =", True),
            ("value in", True),
            # Invalid Parentheses
            ("(feature) = value", True),
            ("(feature = ) value", True),
            ("feature ( = value)", True),
        ],
    )
    def test_parser_syntax(self, event_parser, expression, error_expected):
        if error_expected:
            with pytest.raises(ParseException):
                event_parser.parse(expression)
        else:
            event_parser.parse(expression)

    def test_parser_logic1(self, event_parser):
        expression = "feature = true"
        func = event_parser.parse(expression=expression)

        cur_dict = {"feature": "true"}
        expectation = True
        assert func([cur_dict]) == expectation
        cur_dict = {"feature": "false"}
        expectation = False
        assert (
            func([cur_dict]) == expectation
        ), f'Expression "{expression}" and dictionary {cur_dict} did not evaluate to expected result "{expectation}"'


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
# test value from first dictionary is selected
