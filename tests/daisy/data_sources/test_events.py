# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
from pyparsing import ParseException

from daisy.data_sources import EventHandler


# noinspection PyProtectedMember
@pytest.fixture
def event_parser():
    return EventHandler()._parser


# noinspection PyTypeChecker
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

    @pytest.mark.parametrize(
        "expression,data,expected",
        [
            pytest.param("feature = value", {"feature": "value"}, True, id="=-true"),
            pytest.param(
                "feature = incorrect", {"feature": "value"}, False, id="=-false"
            ),
            pytest.param(
                "value in feature", {"feature": ["value", "other"]}, True, id="in-true"
            ),
            pytest.param("value in feature", {"feature": []}, False, id="in-false"),
        ],
    )
    def test_parser_comparator(self, event_parser, expression, data, expected):
        func = event_parser.parse(expression=expression)
        assert func([data]) == expected

    @pytest.mark.parametrize(
        "expression,expectations,variables",
        [
            ("feature0 = 1", (False, True), 1),
            ("feature0 = 1 and feature1 = 1", (False, False, False, True), 2),
            ("feature0 = 1 or feature1 = 1", (False, True, True, True), 2),
            ("not feature0 = 1", (True, False), 1),
            ("(feature0 = 1)", (False, True), 1),
            ("(feature0 = 1) and feature1 = 1", (False, False, False, True), 2),
            ("feature0 = 1 and (feature1 = 1)", (False, False, False, True), 2),
            ("(feature0 = 1) and (feature1 = 1)", (False, False, False, True), 2),
            ("(feature0 = 1 and feature1 = 1)", (False, False, False, True), 2),
            ("((feature0 = 1) and feature1 = 1)", (False, False, False, True), 2),
            ("not (feature0 = 1 and feature1 = 1)", (True, True, True, False), 2),
            ("not (feature0 = 1 and not feature1 = 1)", (True, True, False, True), 2),
            (
                "feature0 = 1 and (feature1 = 1 or feature2 = 1)",
                (False, False, False, False, False, True, True, True),
                3,
            ),
            (
                "(feature0 = 1 and feature1 = 1) or feature2 = 1",
                (False, True, False, True, False, True, True, True),
                3,
            ),
        ],
    )
    def test_parser_logic(self, event_parser, expression, expectations, variables):
        func = event_parser.parse(expression=expression)
        for permutation in range(2**variables):
            cur_data = {}
            for i in range(variables):
                cur_data["feature" + str(i)] = str(permutation >> variables - 1 - i & 1)

            assert (
                func([cur_data]) == expectations[permutation]
            ), f'Expression "{expression}" and dictionary {cur_data} did not evaluate to expected result "{expectations[permutation]}"'

    @pytest.mark.parametrize(
        "expression",
        [
            "feature = value and",
            "feature = value or feature",
            "feature = value and feature =",
        ],
    )
    def test_parser_incomplete_expression_raises_exception(
        self, event_parser, expression
    ):
        with pytest.raises(ParseException):
            event_parser.parse(expression)

    def test_parser_first_data_occurrence_is_used(self, event_parser):
        func = event_parser.parse("feature = 0")
        assert func([{"feature": "0"}, {"feature": "1"}])
