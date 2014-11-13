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
import numpy as np

from dswont import features as ftr
from dswont import search
from dswont import search_space as sspace
from dswont import util


class WeightUpdateRule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update_weight(self, weigth_vector, positive_vectors,
                      negative_vectors, dubious_vectors,
                      generation):
        return weigth_vector


class PerceptronUpdateRule(WeightUpdateRule):
    def delta(self, weigth_vector, positive_vectors, negative_vectors,
              dubious_vectors, generation):
        delta = np.zeros_like(weigth_vector, float)
        if positive_vectors.size != 0:
            pos_delta = np.average(positive_vectors, axis=0)
            print("Pos vec delta:", pos_delta)
            delta += pos_delta
        if negative_vectors.size != 0:
            neg_delta = np.average(negative_vectors, axis=0)
            print("Neg vec delta:", neg_delta)
            delta -= neg_delta
        #         if dubious_vectors.size != 0:
        #             dub_delta = np.average(dubious_vectors, axis=0)
        #             print("Dub vec delta:", dub_delta)
        #             delta -= 0.2 * dub_delta
        return delta

    def update_weight(self, weigth_vector, positive_vectors,
                      negative_vectors, dubious_vectors, generation):
        delta = self.delta(weigth_vector, positive_vectors,
                           negative_vectors, dubious_vectors, generation)
        result = weigth_vector + delta
        print("Vector update {} + {} -> {}:"
              .format(weigth_vector, delta, result))
        return result


class AggressiveUpdateRule(PerceptronUpdateRule):
    def _loss(self, w, x, y):
        epsilon = 1e-6
        return max(0, 1 + epsilon - y * np.dot(w, x))

    def delta(self, w, positive_vectors, negative_vectors,
              dubious_vectors, generation):
        delta = np.zeros_like(w, float)
        for x in positive_vectors:
            norm_x_squared = np.linalg.norm(x) ** 2
            loss = self._loss(w, x, 1.0)
            delta += x * loss / norm_x_squared
        for x in negative_vectors:
            norm_x_squared = np.linalg.norm(x) ** 2
            loss = self._loss(w, x, -1.0)
            delta -= x * loss / norm_x_squared
        print("Delta: ", delta)
        return delta


class RestartStatesOnFeedbackRule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def restart_states(self, start_state, current_states,
                       positive_states, negative_states,
                       space: sspace.StateSpace):
        return []


class RestartFromScratchRule(RestartStatesOnFeedbackRule):
    def restart_states(self, start_state, current_states,
                       positive_states, negative_states,
                       space: sspace.StateSpace):
        return space.next_states(start_state)


class RestartFromPositiveRule(RestartStatesOnFeedbackRule):
    def restart_states(self, start_state, current_states,
                       positive_states, negative_states,
                       space: sspace.StateSpace):
        return positive_states


class LearningSearch(search.FeatureBasedHeuristicSearch):
    def __init__(self, start: sspace.SearchState, space: sspace.StateSpace,
                 planner: search.SearchPlanner, goal_test,
                 features: ftr.Features, update_rule, restart_rule,
                 weight_vector=None,
                 **params):
        self._update_rule = update_rule
        self._restart_rule = restart_rule
        self._moves = 0
        self._moves_since_restart = 0
        self._updates = 0
        self._restarts = 0
        if weight_vector == None:
            weight_vector = np.zeros(features.n_features(), float)
        super().__init__(start, space, planner, goal_test, features,
                         weight_vector, **params)

    def receive_feedback(self, positive_states, negative_states,
                         dubious_states, restart=True):
        positive_vectors = self._features.compute(positive_states)
        negative_vectors = self._features.compute(negative_states)
        dubious_vectors = self._features.compute(dubious_states)
        self._weight_vector = self._update_rule.update_weight(
            self._weight_vector,
            positive_vectors,
            negative_vectors,
            dubious_vectors,
            self._moves)
        self._updates += 1
        self._cost_fn = search.feature_based_cost_fn(self._features,
                                                     self._weight_vector)
        if restart:
            print("!!!!!!!!!!!!!!!!!!!!!! RESTART !!!!!!!!!!!!!!!!!!!!!!!")
            self.restart(
                self._restart_rule.restart_states(
                    self._start, self.states(),
                    positive_states, negative_states,
                    self._space),
                self._cost_fn)
            self._restarts += 1
            self._moves_since_restart = 0

    def step(self, cost, state):
        self._moves += 1
        self._moves_since_restart += 1
        return super().step(cost, state)


class Teacher(metaclass=abc.ABCMeta):
    def provide_feedback(self, states):
        """Returns the lists of positive and negative states, as a tuple.
        
        """
        positive_states = []
        negative_states = []
        dubious_states = []
        for state in states:
            feedback = self.get_feedback(state)
            if feedback == '+':
                positive_states.append(state)
            elif feedback == '-':
                negative_states.append(state)
            elif feedback == '?':
                dubious_states.append(state)
        return positive_states, negative_states, dubious_states

    @abc.abstractmethod
    def get_feedback(self, state):
        """Returns feedback for the given state.
        
        Interpretation of the returned feedback:
        '+' - positive
        '-' - negative
        '?' - unsure
        '0' - no feedback
        
        """
        return '0'


class SessionCachingTeacher(Teacher):
    def __init__(self, delegate, key):
        self._delegate = delegate
        self._key = key
        self._cache = {}

    def get_feedback(self, state):
        key = self._key(state)
        feedback_value = self._cache.get(key)
        if feedback_value is None:
            feedback_value = self._delegate.get_feedback(state)
            self._cache[key] = feedback_value
        else:
            pass
        return self._cache[key]


class AbstractTeacher(Teacher):
    def __init__(self, get_feedback_fn):
        self._get_feedback = get_feedback_fn

    def get_feedback(self, state):
        return self._get_feedback(state)


def get_feedback_from_stdin(state_to_str):
    def get_feedback(state):
        while True:
            print("Please, provide feedback for the following state:\n{}"
                  .format(state_to_str(state)))
            print(
                "Press 'y' if relevant, 'n' if irrelevant, and 'x' if in doubt")
            answer = util.getch()
            if answer == 'y':
                return '+'
            elif answer == 'n':
                return "-"
            elif answer == 'x':
                return "?"
            elif answer == 'q':
                quit("Bye!")
            else:
                print(
                    "Sorry, '{}' is an invalid input. Please press 'y', 'n' "
                    "or 'x'."
                    .format(answer))

    return get_feedback


class StdInUserFeedbackTeacher(AbstractTeacher):
    def __init__(self, state_to_str):
        self._state_to_str = state_to_str
        super().__init__(get_feedback_from_stdin(state_to_str))


# Bloody mess.
# TODO: Clean this up!
class LearningSearchAlgorithm():
    def __init__(self, search:LearningSearch, teacher:Teacher, state_to_str,
                 alpha=0, steps_no_feedback=0):
        self._search = search
        self._teacher = teacher
        self._state_to_str = state_to_str
        self._search.step(*self._search.next_step())
        self._pos = set()
        self._neg = set()
        self._dub = set()
        self._alpha = alpha
        self._steps_no_feedback = steps_no_feedback
        self._last_feedback = 0
        self._times_feedback_asked = 0
        self._max_moves_across_restarts = 1

    def done(self):
        return self._search.done()

    def _had_feedback(self, node):
        return node in self._pos or node in self._neg or node in self._dub

    def _too_long_without_feedback(self):
        return (
                   self._search._moves - self._last_feedback > self
                   ._steps_no_feedback) \
            and self._search._moves_since_restart >= self\
                   ._max_moves_across_restarts

    def step(self):
        #         print("All states:")
        #         node_seqs = []
        #         for state in self._search.states():
        #             nodes = [action.node() for action in action_seq(state)]
        #             node_seqs.append(sorted(nodes))
        #         print("States (sorted):")
        #         pprint.pprint(sorted(node_seqs))
        cost, state = self._search.next_step()
        restarted = False
        node = state.action().node()
        node_string = self._state_to_str(state)
        depth = state.state().depth(node)
        if self._search._moves % 1 == 0:
            print(
                "Iteration {}, {} restarts, {} moves since restart, "
                "{} feedback points, {} updates, current vector: {}."
                .format(self._search._moves, self._search._restarts,
                        self._search._moves_since_restart,
                        self._times_feedback_asked, self._search._updates,
                        self._search._weight_vector))
            print(
                "Current state: {}(depth {}), {} nodes ({} reached across "
                "restarts), features: {}, utility: {}."
                .format(node_string,
                        depth,
                        state.state().size(),
                        self._max_moves_across_restarts,
                        list(self._search._features.value_map(state).items()),
                        -cost))

        siblings = sspace.get_siblings(state, self._search._space)
        if len(siblings):
            worst_sibling = max(siblings,
                                key=lambda sibling: self._search._cost_fn(
                                    sibling))
            print("Worst sibling: {}, utility: {}"
                  .format(self._state_to_str(worst_sibling),
                          -self._search._cost_fn(worst_sibling)))

        if (-cost < 1 - self._alpha or depth <= 0 \
                    or self._too_long_without_feedback()):
            if self._had_feedback(node):
                print("Already had the feedback for node {}, skipping..".format(
                    node_string))
            else:
                if -cost < 1 - self._alpha:
                    print(
                        "SMALL MARGIN {} for state {}, getting feedback".format(
                            -cost, node_string))
                elif depth <= 1:
                    print(
                        "SMALL DEPTH {} for state {}, getting feedback".format(
                            depth, node_string))
                elif self._too_long_without_feedback():
                    print("TOO MANY STEPS without feedback {}, getting feedback"
                          .format(node_string))
                else:
                    print("THIS CANNOT HAPPEN")
                pos, neg, dub = self._teacher.provide_feedback([state])
                self._times_feedback_asked += 1

                def nodes(states):
                    return [state.action().node() for state in states]

                self._pos.update(nodes(pos))
                self._neg.update(nodes(neg))
                self._dub.update(nodes(dub))
                if state in neg or state in pos:
                    if state in neg or -cost < 1 - self._alpha:
                        restarted = state in neg
                        all_pos = pos
                        all_neg = neg
                        all_dub = dub
                        if len(all_pos) > 0 or len(all_neg) > 0:
                            self._search.receive_feedback(all_pos, all_neg,
                                                          all_dub, restart=(
                                    state in neg))
                        print("New weight vector: {}".format(
                            self._search._weight_vector))
                    else:
                        print(
                            "Positive feedback for the positive node {}, "
                            "proceeding.".format(
                                node_string))
                else:
                    print("No new feedback for {}, proceeding".format(
                        node_string))
                self._last_feedback = self._search._moves
        else:
            print("SUFFICIENT MARGIN {} for the node {}, proceeding".format(
                node_string, -cost))
        self._max_moves_across_restarts = max(self._max_moves_across_restarts,
                                              self._search._moves_since_restart)
        if not restarted:
            self._search.step(cost, state)
        return cost, state