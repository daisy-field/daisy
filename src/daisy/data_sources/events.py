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
    oneOf,
    Suppress,
    Forward,
    Group,
    Optional,
    ParseResults,
    printables,
)
import logging


class Event:
    start_time: datetime
    end_time: datetime
    label: str
    _condition_fn: Callable[[list[dict]], bool]

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        label: str,
        condition_fn: Callable[[list[dict]], bool],
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.label = label
        self._condition_fn = condition_fn

    def evaluate(self, data: list[dict]) -> bool:
        return self._condition_fn(data)


class EventParser:
    _comparators: list[str] = ["=", "in"]
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

        base_word = Word(printables, exclude_chars="[] !\"'<=>\\()")
        bracket_word = Word(printables + " ", exclude_chars="[]!\"'<=>\\()")

        comparator = oneOf(self._comparators)
        boperator = oneOf(self._binary_operators)
        uoperator = oneOf(self._unary_operators)
        lpar = Suppress("(")
        rpar = Suppress(")")
        lbr = Suppress("[")
        rbr = Suppress("]")

        self._parser = Forward()
        word = base_word | lbr + bracket_word + rbr
        operand = Group(word(self._var1) + comparator(self._op) + word(self._var2))
        pars = operand | lpar + self._parser + rpar

        self._parser <<= Group(
            pars(self._var1) + Optional(boperator(self._op) + self._parser(self._var2))
        ) | Group(uoperator(self._op) + pars(self._var1))

    def parse(self, expression: str) -> Callable[[list[dict]], bool]:
        tree = self._parser.parseString(expression, parseAll=True)
        return self._process_child(tree)

    def _process_child(self, tree: ParseResults) -> Optional(
        Callable[[list[dict]], bool]
    ):
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
    ) -> Callable[[list[dict]], bool]:
        if self._op not in dictionary:
            return list(result.values())[0]

        operation = dictionary[self._op]

        if operation in self._binary_operators:
            match operation:
                case "and":
                    return lambda data: result[self._var1](data) and result[self._var2](
                        data
                    )
                case "or":
                    return lambda data: result[self._var1](data) or result[self._var2](
                        data
                    )

                case _:
                    raise NotImplementedError(
                        f"Operation {operation} not supported in EventParser."
                    )

        if operation in self._unary_operators:
            match operation:
                case "not":
                    return lambda data: not result[self._var1](data)

                case _:
                    raise NotImplementedError(
                        f"Operation {operation} not supported in EventParser."
                    )

        if operation in self._comparators:
            match operation:
                case "=":
                    return (
                        lambda data: get_value(dictionary[self._var1][0], data)
                        == dictionary[self._var2][0]
                    )
                case "in":
                    return (
                        lambda data: dictionary[self._var1][0]
                        in get_value(dictionary[self._var2][0], data)
                        if get_value(dictionary[self._var2][0], data)
                        else False
                    )

                case _:
                    raise NotImplementedError(
                        f"Operation {operation} not supported in EventParser."
                    )

        # This should be impossible to reach, since the parser should throw an error already
        raise NotImplementedError(
            f"Operation {operation} not supported in EventParser."
        )


def get_value(feature: str, data: list[dict]) -> object:
    for dictionary in data:
        if feature in dictionary:
            return dictionary[feature]
    raise KeyError(f"Could not find {feature} in {data}")


class EventHandler:
    _parser: EventParser
    _events: list[Event]
    _default_label: str
    _hide_errors: bool

    def __init__(
        self, default_label: str = "benign", hide_errors: bool = False, name: str = ""
    ):
        self._logger = logging.getLogger(name)

        self._parser = EventParser()
        self._events = []
        self._default_label = default_label
        self._hide_errors = hide_errors

    def add_event(
        self, start_time: datetime, end_time: datetime, label: str, condition: str = ""
    ):
        self._logger.debug(f"Adding new event with condition {condition}")
        if condition:
            self._events.append(
                Event(start_time, end_time, label, self._parser.parse(condition))
            )
        else:
            self._events.append(Event(start_time, end_time, label, lambda _: True))

    def process(
        self,
        timestamp: datetime,
        data: list[dict],
        data_point: dict,
        label_feature: str = "label",
    ) -> dict:
        labeled = False
        for event in reversed(self._events):
            if event.start_time < timestamp < event.end_time:
                try:
                    match = event.evaluate(data)
                    if match:
                        data_point[label_feature] = event.label
                        labeled = True
                        break
                except (NotImplementedError, KeyError) as e:
                    self._logger.error(f"Error while evaluating event: {e}")
                    if not self._hide_errors:
                        raise e
        if not labeled:
            data_point[label_feature] = self._default_label

        return data_point
