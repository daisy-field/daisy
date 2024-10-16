# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Author: Jonathan Ackerschewski
Modified: 14.10.24
"""

import sys
from datetime import datetime
from typing import Callable

from pyparsing import (
    ParserElement,
    Word,
    alphas,
    oneOf,
    Suppress,
    Forward,
    Group,
    Optional,
    ParseResults,
)
# TODO add logging


class Event:
    _start_time: datetime
    _end_time: datetime
    _label: str
    _condition_fn: Callable[[dict, dict], bool]

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        label: str,
        condition_fn: Callable[[dict, dict], bool],
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._label = label
        self._condition_fn = condition_fn

    def evaluate(self, meta_information: dict, data_point: dict) -> bool:
        return self._condition_fn(meta_information, data_point)

    def label(
        self,
        timestamp: datetime,
        meta_information: dict,
        data_point: dict,
        label_feature: str = "label",
    ) -> dict:
        if self._start_time < timestamp < self._end_time and self.evaluate(
            meta_information, data_point
        ):
            data_point[label_feature] = self._label
        return data_point


class EventParser:
    _comparators: list[str] = ["=", "in", "<", ">", "<=", ">="]
    _binary_operators: list[str] = ["and", "or"]
    _unary_operators: list[str] = ["not"]

    # identifiers for the parser
    _var1: str = "var1"
    _var2: str = "var2"
    _op: str = "op"

    _parser: ParserElement

    def __init__(self):
        ParserElement.enablePackrat()
        sys.setrecursionlimit(3000)

        word = Word(alphas)
        comparator = oneOf(self._comparators)
        boperator = oneOf(self._binary_operators)
        uoperator = oneOf(self._unary_operators)
        lpar = Suppress("(")
        rpar = Suppress(")")

        self._parser = Forward()
        operand = Group(word(self._var1) + comparator(self._op) + word(self._var2))
        pars = operand | lpar + self._parser + rpar

        self._parser <<= Group(
            pars(self._var1) + Optional(boperator(self._op) + self._parser(self._var2))
        ) | Group(uoperator(self._op) + pars(self._var1))

    def parse(self, expression: str) -> Callable[[dict, dict], bool]:
        tree = self._parser.parseString(expression, parseAll=True)
        return self._process_child(tree)

    def _process_child(self, tree: ParseResults) -> Callable[[dict, dict], bool]:
        result = {}
        if isinstance(tree, ParseResults):
            # Evaluate all children first
            if tree.haskeys():
                for key, _ in reversed(tree.asDict().items()):
                    result[key] = self._process_child(tree.pop())
            else:
                return self._process_child(tree.pop())
        else:
            # On leaf node do nothing
            return None

        return self._create_fn(tree.asDict(), result)

    def _create_fn(
        self, dictionary: dict, result: dict
    ) -> Callable[[dict, dict], bool]:
        if self._op not in dict:
            return list(result.values())[0]

        operation = dict[self._op]

        if operation in self._binary_operators:
            match operation:
                case "and":
                    return lambda meta_information, data_point: result[self._var1](
                        meta_information, data_point
                    ) and result[self._var2](meta_information, data_point)
                case "or":
                    return lambda meta_information, data_point: result[self._var1](
                        meta_information, data_point
                    ) or result[self._var2](meta_information, data_point)

                case _:
                    raise NotImplementedError(
                        f"Operation {operation} not supported in EventParser."
                    )

        if operation in self._unary_operators:
            match operation:
                case "not":
                    return lambda meta_information, data_point: not result[self._var1](
                        meta_information, data_point
                    )

                case _:
                    raise NotImplementedError(
                        f"Operation {operation} not supported in EventParser."
                    )

        if operation in self._comparators:
            pass
        # TODO add functionality for the comparators

        # This should be impossible to reach, since the parser should throw an error already
        raise NotImplementedError(
            f"Operation {operation} not supported in EventParser."
        )
