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
import heapq
import itertools
import numpy as np

from dswont import features as ftr
from dswont import search_space as sspace


################################################################################
## Implementation of the search procedure in some abstract space
################################################################################

class SearchPlanner(metaclass=abc.ABCMeta):
    """Mechanism of scheduling the search states for exploration.

    Normally it is a some sort of [priority] queue.

    """

    @abc.abstractmethod
    def empty(self):
        return True

    @abc.abstractmethod
    def enqueue(self, state: sspace.SearchState, cost: float=0):
        pass

    @abc.abstractmethod
    def dequeue(self) -> sspace.SearchState:
        raise ValueError("Cannot dequeue from an empty planner.")

    @abc.abstractmethod
    def states(self):
        return []


class PriorityQueue(object):
    """Implementation of the priority queue (min-heap) on top of heapq module.

    Supports setting max_size – the maximum size of the queue.
    The maximum value elements are evicted on demand, when the queue becomes
    full. For performance reasons, eviction is not done on every insertion, but
    in batch, when the queue reaches max_buffer_size (default: 2 * max_size).

    """

    def __init__(self, max_size=0, max_buffer_size=0):
        if not max_buffer_size:
            max_buffer_size = 2 * max_size
        assert max_buffer_size >= max_size, \
            "max_buffer_size ({}) must be greater than max_size ({}))" \
                .format(max_buffer_size, max_size)
        self._max_size = max_size
        self._max_buffer_size = max_buffer_size
        self._queue = []
        self._counter = itertools.count()

    def empty(self):
        return len(self._queue) == 0

    def _trim_to_size(self):
        if self._max_size and len(self._queue) > self._max_buffer_size:
            self._queue = heapq.nsmallest(self._max_size, self._queue)
            heapq.heapify(self._queue)

    def put(self, item, cost=0):
        count = next(self._counter)
        entry = (cost, count, item)
        heapq.heappush(self._queue, entry)
        self._trim_to_size()

    def pop(self):
        if len(self._queue) == 0:
            raise IndexError("Cannot pop from an empty PriorityQueue")
        cost, count, item = heapq.heappop(self._queue)
        return cost, item

    def peek(self):
        if len(self._queue) == 0:
            raise IndexError("Cannot peek from an empty PriorityQueue")
        cost, count, item = min(self._queue)
        return cost, item

    def entries(self):
        entry_iterator = ((cost, item) for cost, count, item in
                          sorted(self._queue))
        return itertools.islice(entry_iterator, self._max_size)

    def items(self):
        return (item for cost, item in self.entries())

    def __repr__(self):
        return "PriorityQueue({})".format(self.entries())


class BeamSearchPlanner(SearchPlanner):
    """Implements the scheduling policy for the beam search.

    The states are scheduled according to their cost, with minimum-cost states
    scheduled first. The queue is truncated to contain up to beam_size the best
    states.

    """

    def __init__(self, beam_size):
        assert beam_size >= 1, "Beam size < 1 doesn't make sense: {}".format(
            beam_size)
        self._beam_size = beam_size
        self._queue = PriorityQueue(beam_size)

    def empty(self):
        return self._queue.empty()

    def states(self):
        return self._queue.items()

    def enqueue(self, state: sspace.State, cost: float=0):
        self._queue.put(state, cost)

    def dequeue(self):
        return self._queue.pop()

    def __repr__(self):
        return "BeamSearch({}, {})".format(self._beam_size,
                                           list(self._queue.entries()))


class HeuristicSearch(object):
    """Represents the heuristic search algorithm in some abstract space.

    """

    def __init__(self, start: sspace.SearchState, space: sspace.StateSpace,
                 cost_fn, planner:SearchPlanner, goal_test, **params):
        self._planner = planner
        self._start = start
        self._space = space
        self._goal_test = goal_test
        self._cost_fn = cost_fn
        self._steps = []
        self._last_state = None
        self._planner.enqueue(start)
        self._params = params

    def _param(self, name, default=None):
        return self._params.get(name, default)

    def states(self):
        return self._planner.states()

    def restart(self, states, cost_fn):
        self._cost_fn = cost_fn
        self._steps = []
        self._last_state = None
        while not self._planner.empty():
            self._planner.dequeue()
        for state in states:
            self._planner.enqueue(state, cost_fn(state))

    def next_step(self):
        return self._planner.dequeue()

    def step(self, cost, state):
        if not self._goal_test(state):
            for next_state in self._space.next_states(state):
                cost = self._cost_fn(next_state)
                self._planner.enqueue(next_state, cost)
        self._last_state = state
        if self._param('trace'):
            self._steps.append(cost, state)
        return cost, state

    def done(self) -> bool:
        return (self._last_state is not None
                and self._goal_test(self._last_state)) \
            or self._planner.empty()


def feature_based_cost_fn(features: ftr.Features, weight_vector):
    assert len(features.feature_names()) == len(weight_vector)
    weight_vector = np.array(weight_vector)

    def cost_fn(obj):
        return -np.dot(weight_vector, features.compute_one(obj))

    return cost_fn


class FeatureBasedHeuristicSearch(HeuristicSearch):
    def __init__(self, start: sspace.SearchState, space: sspace.StateSpace,
                 planner: SearchPlanner, goal_test, features: ftr.Features,
                 weight_vector, **params):
        self._features = features
        self._weight_vector = np.array(weight_vector, float)
        cost_fn = feature_based_cost_fn(features, weight_vector)
        super().__init__(start, space, cost_fn,
                         planner, goal_test, **params)