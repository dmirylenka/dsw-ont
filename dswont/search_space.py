# Copyright 2014 University of Trento, Italy.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import abc


################################################################################
## Abstractions of State, Action and SearchState and StateSpace
################################################################################

class State(metaclass=abc.ABCMeta):
    """Represents the point in the a space that we want to navigate.

    A hashable object that should be treated as immutable.
    Works together with Action.

    """
    
    @abc.abstractmethod
    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def __eq__(self, other):
        return self is other
    

class Action(metaclass=abc.ABCMeta):
    """Represents a transformation of the State objects.

    A hashable object that should be treated as immutable.
    Returns the new state without changing the original.
    Works together with State.
    """

    @abc.abstractmethod
    def next(self, state: State) -> State:
        return None

    @abc.abstractmethod
    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def __eq__(self, other):
        return self is other


class SearchState(metaclass=abc.ABCMeta):
    """Represents a point in some abstract space, visited by a search algorithm.

    Contains the information about the point in space being visited, the action
    that brought the algorithm to this point, and the previous SearchState.

    Note that __hash__ and __eq__ are implemented in terms of the previous state
    and the action in case the SearchState is LazyState (in order not to
    'materialize' it without need).

    """

    def __init__(self, state:State, action: Action=None, previous=None):
        self._state = state
        self._action = action
        self._previous = previous

    def state(self) -> State:
        return self._state

    def action(self) -> Action:
        return self._action

    def previous(self):
        return self._previous

    def action_seq(self):
        result = []
        current_state = self
        while current_state._action:
            result.append(current_state._action)
            current_state = current_state._previous
        return list(reversed(result))

    def __hash__(self):
        if self._previous is None:
            # No _action means initial state, and it's ok to 'materialize' it.
            return hash(self.state())
        else:
            return hash((self._previous.state(), self._action))

    def __eq__(self, other):
        if other is None:
            return False
        elif self._previous is None:
            return other._previous is None and self._action == other._action
        else:
            return other._previous is not None and\
                self._previous.state() == other._previous.state() and\
                self._action == other._action

    def __str__(self):
        prev_str = str(self._previous.state()) if self._previous else 'None'
        return "{}->{}".format(prev_str, str(self._action))

    def __repr__(self):
        return "SearchState({}, {}, ...)".format(
            repr(self._state),
            repr(self._action))


class LazyState(SearchState):
    """The SearchState that doesn't compute the contained State until requested.

    Upon calling .state() the current state is computed from the previous state
    and the action, and stored for further invocations.
    """

    def __init__(self, previous: SearchState, action: Action):
        self._previous = previous
        self._action = action
        self._state = None

    def state(self):
        if self._state is None:
            previous_state = self._previous.state()
            self._state = self._action.next(previous_state)
        return self._state

    def __str__(self):
        return "LazyState({}, {})".format(
            str(self._previous.state()),
            str(self._action))

    def __repr__(self):
        return "LazyState({}, {})".format(
            repr(self._previous.state()),
            repr(self._action))


class StateSpace(metaclass=abc.ABCMeta):
    """Represents an abstract search space; defines between-state transitions.

    The derived classes must implement .next_actions(self, state).
    """

    @abc.abstractmethod
    def next_actions(self, state:State):
        """The actions that can be applied to the given state to get new states.
        """
        return []

    def next_states(self, sstate:SearchState):
        """Returns the next possible states for a given state.
        """
        return [LazyState(sstate, action)
                for action in self.next_actions(sstate.state())]


def get_siblings(sstate: SearchState, space: StateSpace):
    """Returns the siblings of the search state in the search space.

    The siblings are the child (next) states of the state's parent (previous)
    state. The siblings of a state include the state itself.

    """
    parent = sstate.previous()
    if parent is None:
        return []
    else:
        return space.next_states(parent)
