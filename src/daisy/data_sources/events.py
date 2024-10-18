# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementation of events used for labeling data. The event handlercan be used by the
user to create events. Each event contains a string with conditions used to determine
whether a data point should be labeled.

Author: Jonathan Ackerschewski
Modified: 14.10.24
"""

import logging
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


class Event:
    """A storage class for an event. Contains the start and end times of the event, the
    label, and the conditions as a function. The condition function should evaluate to
    true, if the data point falls within the event and false otherwise.
    """

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
        """Creates an event, which takes place between start_time and end_time. Data
        points evaluated true by the condition function should be labeled with the
        provided label.

        :param start_time: The start time of the event.
        :param end_time: The end time of the event.
        :param label: The label by which data points should be labeled.
        :param condition_fn: The condition function used to evaluate if a data point
        falls within the event.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.label = label
        self._condition_fn = condition_fn

    def evaluate(self, timestamp: datetime, data: list[dict]) -> bool:
        """Evaluates a single data point using the condition function provided in the
        constructor. The timestamp is used to determine whether the data point falls
        within the events time frame. Additional meta information can be provided by
        passing multiple dictionaries in the data parameter. The dictionaries will be
        searched in the provided order. The first value found will be used for
        comparisons in the condition function.

        :param timestamp: The timestamp of the data point.
        :param data: A single data point and additional meta information. The data
        is searched in the provided order. E.g. if two dictionaries contain the key
        timestamp and the condition function uses this key, then the value of the first
        dictionary will be used.
        """
        if self.start_time <= timestamp <= self.end_time:
            return self._condition_fn(data)
        else:
            return False


class EventParser:
    """This parser is used to parse conditions for events. It takes an expression,
    parses it, and returns a function, which evaluates if a given data point fulfils
    the condition.

    The condition has to follow the following grammar:
    exp := pars + (binary_op + pars)? |
            unary_op + pars
    pars := operand | '(' + exp + ')'
    operand := word + comparator + word
    word := [any character except [] !"'<=>\\()] |
            '[' + [any character except []!"'<=>\\()] + ']'   Note that whitespaces are
                                                              allowed with brackets
    comparator := '=' | 'in'
    binary_op := 'and' | 'or'
    unary_op := 'not'

    For comparators the feature in the dictionary is always expected on the left side
    of the comparator, except with the 'in' operator, where it is expected on the right.

    Some example expressions are:
        ip.addr = 10.1.1.1      When the function is called with a dictionary, it will
                                be searched for the key ip.addr. Its value will be
                                compared to 10.1.1.1
        tcp in protocols        The dictionary will be searched for the key protocols.
                                The function 'tcp in <value of protocols>' will be
                                evaluated.

        ip.addr = 10.1.1.1 and tcp in protocols
        (ip.addr = 10.1.1.1 or ip.addr = 192.168.1.1) and tcp in protocols
        not (ip.addr = 10.1.1.1 or ip.addr = 192.168.1.1) and tcp in protocols

    The returned function can be called using a list of dictionaries. The dictionaries
    will be searched in the provided order and the first occurrence of a feature in one
    of the dictionaries will be used. This can be used to provide meta information about
    a data point additionally to the data point itself.
    """

    # The comparators and operators used by the parser
    _comparators: list[str] = ["=", "in"]
    _binary_operators: list[str] = ["and", "or"]
    _unary_operators: list[str] = ["not"]

    # Identifiers used inside the parser to mark specific tokens
    _var1: str = "var1"
    _var2: str = "var2"
    _op: str = "op"

    _parser: ParserElement

    def __init__(self):
        """Creates the grammar for the conditions. The specific grammar can be seen
        in the description of this class.
        """
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
        """Parses the given expression and returns a function that evaluates data points
        passed to it. This method can raise a parse error if the expression is invalid.

        :param expression: The expression (condition) to parse
        """
        tree = self._parser.parseString(expression, parseAll=True)
        return self._process_child(tree)

    def _process_child(self, tree: ParseResults) -> Optional(
        Callable[[list[dict]], bool]
    ):
        """Recursively processes the provided parse tree and returns a function that
        evaluates the data points passed to it.

        :param tree: The parse tree to process
        """
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
        """Creates a function based on the provided input. The provided dictionary
        should contain any of the keys _op, _var1, and _var2 from this class.
        When the key _op is any of the binary operators, _var1 and _var2 have to be
        present in the result dictionary and contain a function. If the operator is
        any unary operator, only _var1 needs to be present in the result dictionary.
        If the operator is any comparator, the result dictionary is ignored and the
        dictionary needs to contain the _var1 and _var2 keys.
        The returned function can raise key errors if the dictionaries passed to it
        do not contain the keys required by the generated function.

        :param dictionary: The dictionary used to evaluate the function to generate
        :param result: A dictionary containing results of previous runs of this function
        """
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

        raise NotImplementedError(
            f"Operation {operation} not supported in EventParser."
        )


def get_value(feature: str, data: list[dict]) -> object:
    """Iterates through the provided dictionaries and returns the value of the first
    occurrence of the provided feature.

    :param feature: The feature to find
    :param data: A list of dictionaries to search for the feature
    """
    for dictionary in data:
        if feature in dictionary:
            return dictionary[feature]
    raise KeyError(f"Could not find {feature} in {data}")


class EventHandler:
    """The event handler is used to create events and label data points. Events can be
    added to the class using the `add_event` method. The events are matched to the
    data points in the added order. The first matching event will be used to label
    the data point.
    """

    _parser: EventParser
    _events: list[Event]
    _default_label: str
    _hide_errors: bool

    def __init__(
        self, default_label: str = "benign", hide_errors: bool = False, name: str = ""
    ):
        """Creates an event handler used to label data points. The handler is provided
        with a default label to use and if errors should be hidden. The events use a
        condition function which evaluates each data point. This function can throw
        key errors if the provided data point does not contain features used by the
        condition function. The hide errors parameter catches the errors and only
        prints the error in the logs to prevent a long-running code from crashing.
        Since it cannot be decided if the data point fulfills the event in this case,
        the data point will be labeled with an error.

        :param default_label: This label is used when no event matches the data point
        :param hide_errors: Catches exceptions in the process method and only prints
        errors to the logs instead.
        """
        self._logger = logging.getLogger(name)

        self._parser = EventParser()
        self._events = []
        self._default_label = default_label
        self._hide_errors = hide_errors

    def add_event(
        self, start_time: datetime, end_time: datetime, label: str, condition: str = ""
    ):
        """Adds an event to the event handler. The events will be evaluated in the
        order they are provided. Each event has a start and end time, a label that will
        be used to label the data point and an optional condition. The condition is
        a string and has to follow a certain grammar, which is explained in detail in
        the EventParser documentation.

        :param start_time: The start time of the event
        :param end_time: The end time of the event
        :param label: The label of the event
        :param condition: The condition of the event
        """
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
        error_label: str = "error",
    ) -> dict:
        """Iterates through all events and checks for each event if it applies to
        the provided data point. If it does, the data point will be labeled with the
        label provided by the event. If no event matches the data point, it will be
        labeled with the default label. The timestamp is used to determine if the data
        point is within the specified time range of the events. The data parameter can
        contain multiple dictionaries, but at least should contain the data point.
        If an error occurs with the condition of an event, the data point will be
        labeled with the error label.

        :param timestamp: The timestamp of the data point
        :param data: The data point and meta information
        :param data_point: The data point, used for labeling
        :param label_feature: The feature in the data point where the label will be
        placed
        :param error_label: The label to use in case of an error
        """
        labeled = False
        for event in self._events:
            try:
                match = event.evaluate(timestamp, data)
                if match:
                    data_point[label_feature] = event.label
                    labeled = True
                    break
            except (NotImplementedError, KeyError) as e:
                self._logger.error(f"Error while evaluating event: {e}")
                data_point[label_feature] = error_label
                labeled = True
                if not self._hide_errors:
                    raise e
                break
        if not labeled:
            data_point[label_feature] = self._default_label

        return data_point
