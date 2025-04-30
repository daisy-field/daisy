# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Events used to labeling data. The event handler can be used by the user to create
events. Each event contains a string with conditions used to determine whether a data
point should be labeled.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.24
"""

import logging
import sys
from datetime import datetime
from typing import Callable, Self, Optional

import pyparsing as pp


class EventParser:
    """Parser for conditions of events. It takes an expression, parses it, and returns
    a function, which evaluates if a given data point fulfils the condition.

    The condition has to follow the following grammar:
    exp := pars + (binary_op + pars)? |
            unary_op + pars
    pars := operand | '(' + exp + ')'
    operand := cast + comparator + cast
    cast := cast_op + '!' + word | word
    word := [any character except [] !"'<=>\\()] |
            '[' + [any character except []!"'<=>\\()] + ']'   Note that whitespaces are
                                                              allowed with brackets
    comparator := '=' | 'in' | '<' | '>' | '<=' | '>='
    binary_op := 'and' | 'or'
    unary_op := 'not'
    cast_op := 's' | 'i' | 'f' | 't'

    For comparators, the feature in the dictionary is always expected on the left side
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
        port = i!22             The 22 will be cast to integer before comparison.

    The returned function can be called using a list of dictionaries. The dictionaries
    will be searched in the provided order and the first occurrence of a feature in one
    of the dictionaries will be used. This can be used to provide meta information about
    a data point additionally to the data point itself.

    Values in the expression can be cast to different types to support proper comparison.
    The supported castings are string (s), integer (i), float (f), and timestamp(t).
    For timestamp casting, the following formats are supported:
    time since epoch
    DD.MM.YY-HH:MM:SS.ffffff
    DD.MM.YY-HH:MM:SS
    DD.MM.YY HH:MM:SS.ffffff
    DD.MM.YY HH:MM:SS

    For all cases, except the time since epoch, the timezone can be specified using
    an offset from UTC in the form +/-HHMM[SS[.ffffff]], e.g. -0200, +0000. If no
    offset is specified, UTC is assumed.
    """

    # The comparators and operators used by the parser
    _comparators: list[str] = ["=", "in", "<=", ">=", "<", ">"]
    _binary_operators: list[str] = ["and", "or"]
    _unary_operators: list[str] = ["not"]
    _casts: str = "sift"

    # Identifiers used inside the parser to mark specific tokens
    _var1: str = "var1"
    _var2: str = "var2"
    _op: str = "op"
    _cast: str = "cast"
    _word: str = "word"

    _parser: pp.ParserElement

    def __init__(self):
        """Creates the grammar for the conditions. The specific grammar can be seen
        in the description of this class.
        """
        pp.ParserElement.enablePackrat()
        sys.setrecursionlimit(3000)

        base_word = pp.Word(pp.printables, exclude_chars="[] !\"'<=>\\()")
        bracket_word = pp.Word(pp.printables + " ", exclude_chars="[]!\"'<=>\\()")
        cast_types = pp.Char(self._casts)

        comparator = pp.oneOf(self._comparators)
        boperator = pp.oneOf(self._binary_operators)
        uoperator = pp.oneOf(self._unary_operators)
        cast_op = pp.oneOf("!")
        lpar = pp.Suppress("(")
        rpar = pp.Suppress(")")
        lbr = pp.Suppress("[")
        rbr = pp.Suppress("]")

        self._parser = pp.Forward()
        word = base_word | lbr + bracket_word + rbr
        cast = pp.Group(cast_types(self._cast) + cast_op + word(self._word)) | word
        operand = pp.Group(cast(self._var1) + comparator(self._op) + cast(self._var2))
        pars = operand | lpar + self._parser + rpar

        self._parser <<= pp.Group(
            pars(self._var1)
            + pp.Optional(boperator(self._op) + self._parser(self._var2))
        ) | pp.Group(uoperator(self._op) + pars(self._var1))

    def parse(self, expression: str) -> Callable[[dict], bool]:
        """Parses the given expression and returns a function that evaluates data points
        passed to it. This method can raise a parse error if the expression is invalid.

        :param expression: Expression (condition) to parse.
        :raises ParseError: Any condition/expression does not follow parser's grammar.
        """
        tree = self._parser.parseString(expression, parseAll=True)
        return self._process_child(tree, expression)

    def _process_child(
        self, tree: pp.ParseResults, expression: str
    ) -> Optional[Callable[[dict], bool]]:
        """Recursively processes the provided parse tree and returns a function that
        evaluates the data points passed to it.

        :param tree: The parse tree to process
        :param expression: The original expression. Passed for error handling purposes
        """
        result = {}
        if isinstance(tree, pp.ParseResults):
            # Evaluate all children first
            if tree.haskeys():
                for key, _ in reversed(tree.asDict().items()):
                    result[key] = self._process_child(tree.pop(), expression)
            else:
                return self._process_child(tree.pop(), expression)
        else:
            # On leaf node do nothing
            return None

        return self._create_fn(tree.asDict(), result, expression)

    def _create_fn(
        self, dictionary: dict, result: dict, expression: str
    ) -> Callable[[dict], bool]:
        """Creates a function based on the provided input. The provided dictionary
        should contain any of the keys _op, _var1, and _var2 from this class.
        When the key _op is any of the binary operators, _var1 and _var2 have to be
        present in the result dictionary and contain a function. If the operator is
        any unary operator, only _var1 needs to be present in the result dictionary.
        If the operator is any comparator, the result dictionary is ignored and the
        dictionary needs to contain the _var1 and _var2 keys.
        The returned function can raise key errors if the dictionaries passed to it
        do not contain the keys required by the generated function.

        :param dictionary: Dictionary used to evaluate the function to generate.
        :param result: Dictionary containing results of previous runs of this function.
        :param expression: The original expression. Passed for error handling purposes
        :raises NotImplementedError: Operators used are unsupported by this class.
        Only happens when class has been modified/overridden.
        """
        if self._op not in dictionary:
            return list(result.values())[0]

        operation = dictionary[self._op]

        if operation in self._binary_operators:
            return self._create_binary_fn(result, operation, expression)

        if operation in self._unary_operators:
            return self._create_unary_fn(result, operation, expression)

        if operation in self._comparators:
            return self._create_comparator_fn(dictionary, operation, expression)

        raise NotImplementedError(
            f"Operation '{operation}' not supported in EventParser for "
            f"condition '{expression}'."
        )

    def _create_binary_fn(
        self, result: dict, operation: str, expression: str
    ) -> Callable[[dict], bool]:
        """Creates the binary function part. The results of previous runs of the
        create_fn method are used in this function to combine them using binary
        functions (and, or operators).

        :param result: Dictionary containing the results of previous runs.
        :param operation: Operation to perform.
        :param expression: The original expression. Passed for error handling purposes
        :raises NotImplementedError: Operators used are unsupported by this class.
        Only happens when class has been modified/overridden.
        """
        match operation:
            case "and":
                return lambda data: result[self._var1](data) and result[self._var2](
                    data
                )
            case "or":
                return lambda data: result[self._var1](data) or result[self._var2](data)
            case _:
                raise NotImplementedError(
                    f"Operation '{operation}' not supported in EventParser for "
                    f"condition '{expression}'."
                )

    def _create_unary_fn(
        self, result: dict, operation: str, expression: str
    ) -> Callable[[dict], bool]:
        """Creates the unary function part. The results of previous runs of the
        create_fn method are used in this function to combine them using unary
        functions (not operator).

        :param result: Dictionary containing the results of previous runs
        :param operation: Operation to perform.
        :param expression: The original expression. Passed for error handling purposes
        :raises NotImplementedError: Operators used are unsupported by this class.
        Only happens when class has been modified/overridden.
        """
        match operation:
            case "not":
                return lambda data: not result[self._var1](data)
            case _:
                raise NotImplementedError(
                    f"Operation '{operation}' not supported in EventParser for "
                    f"condition '{expression}'."
                )

    def _create_comparator_fn(
        self, dictionary: dict, operation: str, expression: str
    ) -> Callable[[dict], bool]:
        """Creates the comparisons of the function. The dictionary should contain
        a feature, operation, and value. Based on the comparator, a function is
        returned, which performs the desired comparison.

        :param dictionary: Dictionary containing feature and value to use for the
        generated function.
        :param operation: Operation to perform.
        :param expression: The original expression. Passed for error handling purposes
        :raises NotImplementedError: Operators used are unsupported by this class.
        Only happens when class has been modified/overridden.
        """
        if operation == "in":
            self._check_feature(dictionary[self._var2][0], expression)
            value = self._get_value(dictionary[self._var1][0])
            return (
                lambda data: value in data[dictionary[self._var2][0]]
                if data[dictionary[self._var2][0]]
                else False
            )

        feature = dictionary[self._var1][0]
        self._check_feature(feature, expression)
        value = self._get_value(dictionary[self._var2][0])

        match operation:
            case "=":
                return lambda data: data[feature] == value
            case "<":
                return lambda data: data[feature] < value
            case ">":
                return lambda data: data[feature] > value
            case "<=":
                return lambda data: data[feature] <= value
            case ">=":
                return lambda data: data[feature] >= value
            case _:
                raise NotImplementedError(
                    f"Operation '{operation}' not supported in EventParser for "
                    f"condition '{expression}'."
                )

    def _check_feature(self, feature: object, expression: str):
        """Checks the given feature for errors.

        :param feature: The feature to check.
        :param expression: The original expression. Passed for error handling purposes
        :throws RuntimeError: If the feature is invalid.
        """
        if isinstance(feature, dict):
            raise RuntimeError(
                f"Tried casting feature '{feature[self._word][0]}' using "
                f"casting operator '{feature[self._cast]}' in event condition "
                f"'{expression}'."
            )

    def _get_value(self, dictionary: dict) -> object:
        """Gets the value from the dictionary. If a cast operation is defined,
        the value will be cast to the desired type.

        :param dictionary: The dictionary to get the value from.
        :return: The (optionally) cast value
        :raises NotImplementedError: Operators used are unsupported by this class.
        :raises ValueError: If a cast operation fails
        """
        if self._cast not in dictionary:
            return dictionary
        match dictionary[self._cast]:
            case "i":
                try:
                    return int(dictionary[self._word][0])
                except ValueError:
                    raise ValueError(
                        f"Casting '{dictionary[self._word][0]}' to integer failed."
                    )
            case "s":
                try:
                    return str(dictionary[self._word][0])
                except ValueError:
                    raise ValueError(
                        f"Casting '{dictionary[self._word][0]}' to string failed."
                    )
            case "f":
                try:
                    return float(dictionary[self._word][0])
                except ValueError:
                    raise ValueError(
                        f"Casting '{dictionary[self._word][0]}' to float failed."
                    )
            case "t":
                try:
                    return _cast_to_timestamp(dictionary[self._word][0])
                except ValueError:
                    raise ValueError(
                        f"Casting '{dictionary[self._word][0]}' to datetime failed."
                    )

        raise NotImplementedError(
            f"The given cast '{dictionary[self._cast]}' is not implemented."
        )


class EventHandler:
    """Event handler used to create events and automatically label data points. Events
    can be added to the class using the add_event() method. The events are matched to
    data points in added order, i.e. first matching event will be used to label
    a given data point.
    """

    _parser: EventParser
    _event_label_pairs: list[tuple[str, Callable[[dict], bool]]]
    _default_label: str
    _label_feature: str
    _error_label: str
    _hide_errors: bool

    def __init__(
        self,
        default_label: str = "benign",
        label_feature: str = "label",
        error_label: str = "error",
        hide_errors: bool = False,
        name: str = "EventHandler",
        log_level: int = None,
    ):
        """Creates an event handler used to label data points.

        :param default_label: Label used when no event matches data point.
        :param label_feature: Feature in data point for which label will be set.
        :param error_label: Error label to use if an error is encountered during
        processing.
        :param hide_errors: Catches any key errors occurring in process() method when
        encountering data points not containing features used by the conditions of
        an event, only printing them out in the logs instead of exciting, labeling
        the data point as erroneous.
        :param name: Name of event handler for logging purposes.
        :param log_level: Logging level for logging purposes.
        """
        self._logger = logging.getLogger(name)
        if log_level:
            self._logger.setLevel(log_level)

        self._parser = EventParser()
        self._events = []
        self._event_label_pairs = []
        self._default_label = default_label
        self._label_feature = label_feature
        self._error_label = error_label
        self._hide_errors = hide_errors

    def add_event(self, label: str, condition: str) -> Self:
        """Adds an event to the event handler. The events will be evaluated in the
        order they are provided. Each event has a label that will be used to label data
        points that fall under that event, and a condition. The condition is
        a string and has to follow a certain grammar:

        exp := pars + (binary_op + pars)? |
                unary_op + pars
        pars := operand | '(' + exp + ')'
        operand := cast + comparator + cast
        cast := cast_op + '!' + word | word
        word := [any character except [] !"'<=>\\()] |
                '[' + [any character except []!"'<=>\\()] + ']'   Note that whitespaces
                                                                  are allowed with
                                                                  brackets
        comparator := '=' | 'in' | '<' | '>' | '<=' | '>='
        binary_op := 'and' | 'or'
        unary_op := 'not'
        cast_op := 's' | 'i' | 'f' | 't'

        For comparators, the feature in the dictionary is always expected on the left
        side of the comparator, except with the 'in' operator, where it is expected
        on the right.

        Some example expressions are:
        ip.addr = 10.1.1.1      When the function is called with a dictionary, it will
                                be searched for the key ip.addr. Its value will be
                                compared to 10.1.1.1
        tcp in protocols        The dictionary will be searched for the key protocols.
                                The function 'tcp in <value of protocols>' will be
                                evaluated.

        Concatenation examples are:
        ip.addr = 10.1.1.1 and tcp in protocols
        (ip.addr = 10.1.1.1 or ip.addr = 192.168.1.1) and tcp in protocols
        not (ip.addr = 10.1.1.1 or ip.addr = 192.168.1.1) and tcp in protocols

        Values in the expression can be cast to different types to support proper
        comparison.
        The supported castings are string (s), integer (i), float (f), and timestamp(t).
        For timestamp casting, the following formats are supported:
        time since epoch
        DD.MM.YY-HH:MM:SS.ffffff
        DD.MM.YY-HH:MM:SS
        DD.MM.YY HH:MM:SS.ffffff
        DD.MM.YY HH:MM:SS

        For all cases, except the time since epoch, the timezone can be specified using
        an offset from UTC in the form +/-HHMM[SS[.ffffff]], e.g. -0200, +0000. If no
        offset is specified, UTC is assumed.

        The returned function can be called using a dictionary.

        :param label: Label of event.
        :param condition: Condition(s) data points have to fulfill for this event.
        """
        self._logger.debug(f"Adding new event with condition {condition}")
        if not condition:
            raise ValueError("Condition cannot be empty")
        self._event_label_pairs.append((label, self._parser.parse(condition)))

        return self

    def evaluate(self, data_point: dict) -> dict:
        """Iterates through all events and checks for each event if it applies to
        the provided data point. If it does, the data point will be labeled with the
        label provided by the event. If no event matches the data point, it will be
        labeled with the default label.

        :param data_point: Data point to label.
        preference over data point when checking conditions.
        processing and errors are suppressed.
        :return: Labelled data point.
        :raises KeyError: Data point does not contain feature used by a conditions and
        errors are not suppressed, i.e. redirected to log + data point is assigned
        the error label.
        """
        for label, condition in self._event_label_pairs:
            try:
                if condition(data_point):
                    data_point[self._label_feature] = label
                    return data_point
            except KeyError as e:
                if not self._hide_errors:
                    raise e
                self._logger.error(f"Error while evaluating event: {e}")
                data_point[self._label_feature] = self._error_label
                return data_point
        data_point[self._label_feature] = self._default_label
        return data_point


def _cast_to_timestamp(exp: str) -> float:
    """Casts the given string to a timestamp. The following formats are supported:

    DD.MM.YYTHH:MM:SS.ffffff
    DD.MM.YYTHH:MM:SS
    DD.MM.YY HH:MM:SS.ffffff
    DD.MM.YY HH:MM:SS
    time since epoch

    For all cases, except the time since epoch, the timezone can be specified using
    an offset from UTC in the form +/-HHMM[SS[.ffffff]], e.g. -0200, +0000. If no
    offset is specified, UTC is assumed.

    :param exp: The expression to cast
    :return: a datetime
    :raises ValueError: If the given expression does not match any of
    the supported formats.
    """
    try:
        return float(exp)
    except ValueError:
        pass
    try:
        return datetime.strptime(exp, "%d.%m.%yT%H:%M:%S.%f%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp, "%d.%m.%y %H:%M:%S.%f%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp + "+0000", "%d.%m.%yT%H:%M:%S.%f%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp + "+0000", "%d.%m.%y %H:%M:%S.%f%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp, "%d.%m.%yT%H:%M:%S%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp, "%d.%m.%y %H:%M:%S%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp + "+0000", "%d.%m.%yT%H:%M:%S%z").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(exp + "+0000", "%d.%m.%y %H:%M:%S%z").timestamp()
    except ValueError:
        pass
    raise ValueError(f"Given date {exp} does not match any date or time pattern.")
