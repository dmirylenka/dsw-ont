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
import collections
import numpy as np
import operator
import pprint

from dswont import features as ftr
from dswont import search
from dswont import search_space as sspace
from dswont import util


##==============================================================================
## Definition of the main data types used in the learning search algorithm.
##==============================================================================

class UserFeedback(metaclass=abc.ABCMeta):
    """'Marker interface' for feedback that the can be given to the algorithm.

    """

    def __init__(self, points):
        self.points = points

    @classmethod
    @abc.abstractmethod
    def query(cls, points):
        """Returns a query -- empty feedback without any user input.

        """
        return cls(points)

    @abc.abstractmethod
    def map_features(self, features: ftr.Features):
        """Transforms all feedback points into their feature representation.

        """
        features_feedback = self.copy()
        features_feedback._points = features.compute(self.points)
        return features_feedback

    @abc.abstractmethod
    def copy(self):
        """Returns a copy of the feedback object.

        When providing the feedback for the query, teachers should .copy() the
        query object and return the copy with the updated feedback information.

        """
        pass


class LearningSearchState(object):
    """The state of the learning search algorithm.

    This is the major data API between the components of the learning search.

    """

    def __init__(self, iteration=0, restarts=[], feedback=None, weights=None):
        self._iteration = iteration
        self._restarts = restarts
        self._feedback = feedback or collections.OrderedDict()
        self._weights = weights or collections.OrderedDict()

    def copy(self):
        return LearningSearchState(
            self._iteration,
            self._restarts.copy(),
            self._feedback.copy(),
            self._weights
        )

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def restarts(self):
        return self._restarts

    @property
    def feedback(self):
        return self._feedback

    @property
    def weights(self):
        return self._weights

    def new_iteration(self):
        old_state = self.copy()
        self._iteration += 1
        return old_state

    def record_feedback(self, feedback: UserFeedback):
        assert self._iteration not in self._feedback
        self._feedback[self._iteration] = feedback
        return self

    def record_restart(self):
        self._restarts.append(self._iteration)
        return self

    def record_update(self, weights):
        assert self._iteration not in self._weights
        self._weights[self._iteration] = weights
        return self

    @property
    def n_restarts(self):
        return len(self._restarts)

    @property
    def moves_since_restart(self):
        last_restart = self._restarts[-1] if self._restarts else 0
        return self._iteration - last_restart

    @property
    def n_feedback(self):
        return len(self._feedback)

    @property
    def n_updates(self):
        return len(self._weights)


class MultiBinaryFeedback(UserFeedback):
    """Feedback of the form {positive, negative, not_sure} for multiple states.

    """

    POSITIVE_FEEDBACK = 1
    NEGATIVE_FEEDBACK = -1
    NOT_SURE_FEEDBACK = 0
    NO_FEEDBACK = None

    def __init__(self, points, labels):
        self.points = points
        self.labels = labels

    @classmethod
    def query(cls, points):
        empty_labels = [MultiBinaryFeedback.NO_FEEDBACK] * len(points)
        return cls(points, empty_labels)

    @property
    def positive_points(self):
        return [point for point, label in zip(self.points, self.labels)
                      if label == MultiBinaryFeedback.POSITIVE_FEEDBACK]

    @property
    def negative_points(self):
        return [point for point, label in zip(self.points, self.labels)
                      if label == MultiBinaryFeedback.NEGATIVE_FEEDBACK]

    def map_features(self, features: ftr.Features):
        features_feedback = self.copy()
        features_feedback.points = features.compute(self.points)
        return features_feedback

    def copy(self):
        return MultiBinaryFeedback(self.points.copy(),
                                   self.labels.copy())


class MultiSelectFeedback(UserFeedback):
    """Feedback specifying selection of one set of points from a larger set.

    User selects the set of nodes A from a larger set B.
    It is assumed that points in the set A a preferred over the points in the
    set B\A. It is also assumed that the points in A are all positive, while the
    points in B\A may be positive or negative.

    """

    def __init__(self, preferred_points, other_points):
        self.preferred = preferred_points
        self.other = other_points

    @classmethod
    def query(cls, points):
        return cls([], points)

    def map_features(self, features: ftr.Features):
        feature_feedback = self.copy()
        feature_feedback.preferred = features.compute(self.preferred)
        feature_feedback.other = features.compute(self.other)
        return feature_feedback

    def copy(self):
        return MultiSelectFeedback(self.preferred.copy(),
                                   self.other.copy())


##==============================================================================
## Definition of the abstract components of the learning search algorithm.
##==============================================================================


class QueryingCondition(metaclass=abc.ABCMeta):
    """Decides if the feedback should be asked of the user.

    """

    @abc.abstractmethod
    def feedback_required(self,
                          learning_state: LearningSearchState,
                          current_cost_state_pair: (float, sspace.SearchState),
                          next_cost_state_pairs) -> UserFeedback:
        """Returns the feedback that has to be asked of the user, or None.

        """
        pass


class WeightUpdateCondition(metaclass=abc.ABCMeta):
    """Decides if the feature weights should be updated.

    """

    @abc.abstractmethod
    def should_update(self,
                      learning_state: LearningSearchState,
                      current_cost_state_pair: (float, sspace.SearchState),
                      next_cost_state_pairs,
                      feedback: UserFeedback):
        return False


class WeightUpdateRule(metaclass=abc.ABCMeta):
    """Specifies how the weights must be updated upon getting feedback.

    """
    @abc.abstractmethod
    def update_weights(self, weight_vector,
                      feedback: UserFeedback,
                      learning_state: LearningSearchState):
        return weight_vector


class RestartOnFeedbackRule(metaclass=abc.ABCMeta):
    """Decides if the search algorithm should be restarted (e.g. on mistake).

    Also decides which states the algorithm should restart from.
    
    """
    @abc.abstractmethod
    def restart_states(self,
                       learning_state: LearningSearchState,
                       start_state: sspace.SearchState,
                       current_cost_state_pair: (float, sspace.SearchState),
                       next_cost_state_pairs,
                       feedback: UserFeedback):
        """Returns the states from which to restart, if restart is required.

        Returns None otherwise.

        """
        return None


class Teacher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_feedback(self, query: UserFeedback) -> UserFeedback:
        pass


##==============================================================================
## Definition of the learning search algorithm itself.
##==============================================================================

class LearningSearch(search.FeatureBasedHeuristicSearch):
    """A search algorithm that learns from the feedback.

    """
    def __init__(self,
                 start: sspace.SearchState,
                 space: sspace.StateSpace,
                 planner: search.SearchPlanner,
                 goal_test: 'sspace.SearchState -> bool',
                 features: ftr.Features,
                 weight_vector,
                 querying_condition: QueryingCondition,
                 teacher: Teacher,
                 update_condition: WeightUpdateCondition,
                 update_rule: WeightUpdateRule,
                 restart_rule: RestartOnFeedbackRule,
                 state_to_str,
                 **params):
        super().__init__(start, space, planner, goal_test, features,
                         weight_vector, **params)
        self._querying_condition = querying_condition
        self._teacher = teacher
        self._update_condition = update_condition
        self._update_rule = update_rule
        self._restart_rule = restart_rule
        self._learning_state = LearningSearchState()
        self._state_to_str = state_to_str
        super().step()

    def report_iteration(self,
                         current_cost: float,
                         current_state: sspace.SearchState,
                         learning_state: LearningSearchState,
                         next_cost_state_pairs):
        print(
            "Iteration {}, {} restarts, {} moves since restart, "
            "{} feedback points, {} updates, current vector: {}."
            .format(learning_state.iteration,
                    learning_state.n_restarts,
                    learning_state.moves_since_restart,
                    learning_state.n_feedback,
                    learning_state.n_updates,
                    util.format_nums(self._weight_vector, 5)))
        current_node = current_state.action().node()
        node_string = self._state_to_str(current_state)
        node_depth = current_state.state().depth(current_node)
        print(
            "Current state: {}(depth {}), {} nodes, "
            "features: {}, score: {}."
            .format(node_string,
                    node_depth,
                    current_state.state().size(),
                    list(self._features.value_map(current_state).items()),
                    -current_cost))


    def step(self) -> (float, sspace.SearchState):
        current_cost, current_state = self._planner.peek()
        previous_learning_state = self._learning_state.new_iteration()
        if not self._goal_test(current_state):
            next_cost_state_pairs = self.next_cost_state_pairs()
            self.report_iteration(current_cost, current_state,
                                  previous_learning_state,
                                  next_cost_state_pairs)
            query = self._querying_condition.feedback_required(
                previous_learning_state,
                (current_cost, current_state),
                next_cost_state_pairs
            )
            if query is not None:
                feedback = self._teacher.get_feedback(query)
                self._learning_state.record_feedback(feedback)
                # TODO: check if update is needed (if weights actually change)
                update_needed = self._update_condition.should_update(
                    previous_learning_state,
                    (current_cost, current_state),
                    next_cost_state_pairs,
                    feedback
                )
                if update_needed:
                    updated_weights = self._update_rule.update_weights(
                        self._weight_vector,
                        feedback.map_features(self._features),
                        previous_learning_state
                    )
                    print("Updated vector:\n{} + {} -> {}"
                          .format(util.format_nums(self._weight_vector, 5),
                                  util.format_nums(updated_weights -
                                                   self._weight_vector, 5),
                                  util.format_nums(updated_weights, 5)))
                    self.weight_vector = updated_weights
                    self._learning_state.record_update(updated_weights)
                restart_states = self._restart_rule.restart_states(
                    previous_learning_state,
                    self._start,
                    (current_cost, current_state),
                    next_cost_state_pairs,
                    feedback
                )
                if restart_states is not None:
                    print("\n!!! RESTART CONDITION !!!\n")
                    self.restart(restart_states)
                    self._learning_state.record_restart()
        super().step()
        return current_cost, current_state


##==============================================================================
## Definition of the specific components of the learning search algorithm.
##==============================================================================

##------------------------------------------------------------------------------
## Querying conditions

class SmallMarginOfCurrentNodeBinaryQueryingCondition(QueryingCondition):

    """Queries user for feedback on the dequeued state when its score is small.

    The score is defined as minus cost of the state. The score is considered
    small when it is less than 1 - alpha, where alpha is the input parameter.

    """
    def __init__(self, alpha):
        self.alpha = alpha

    def feedback_required(self,
                          learning_state: LearningSearchState,
                          current_cost_state_pair: (float, sspace.SearchState),
                          next_cost_state_pairs) -> MultiBinaryFeedback:
        cost, state = current_cost_state_pair
        if -cost < 1 - self.alpha:
            return MultiBinaryFeedback.query([state])
        else:
            return None


class ScoreDecreaseBinaryQueryingCondition(QueryingCondition):

    """Asks for feedback when the score decreases from the current to next node.

    The score is defined as minus cost of the state.
    The feedback is asked for when (score(best_child) < score(parent) + gamma).

    """

    def __init__(self, gamma):
        self.gamma = gamma

    def feedback_required(self,
                          learning_state: LearningSearchState,
                          current_cost_state_pair: (float, sspace.SearchState),
                          next_cost_state_pairs) -> MultiBinaryFeedback:
        cost, state = current_cost_state_pair
        best_cost, best_node = next_cost_state_pairs[0]
        worst_cost, worst_node = next_cost_state_pairs[-1]
        print("Current node {}, score {:.4f}".format(state, -cost))
        print("Best next node {}, score {:.4f}".format(best_node, -best_cost))
        print("Worst next node {}, score {:.4f}".format(worst_node, -worst_cost))
        if -best_cost < -cost + self.gamma:
            print("\n!!! QUERYING CONDITION !!!\n")
            other_nodes = [node for _, node in next_cost_state_pairs[1:-1]]
            np.random.seed(0)
            some_more_nodes = np.random.choice(other_nodes, 3, replace=False)
            all_nodes = [best_node, worst_node] + list(some_more_nodes)
            return MultiSelectFeedback.query(all_nodes)
        else:
            return None


##------------------------------------------------------------------------------
## Update conditions

class PerceptronMistakeOnCurrentNodeUpdateCondition(WeightUpdateCondition):
    """Update if preceptron prediction on the current node contradicts feedback.

    """

    def should_update(self,
                      learning_state: LearningSearchState,
                      current_cost_state_pair: (float, sspace.SearchState),
                      next_cost_state_pairs,
                      feedback: MultiBinaryFeedback):
        cost, node = current_cost_state_pair
        score = -cost
        node_is_pos = node in feedback.positive_points
        node_is_neg = node in feedback.negative_points
        assert not node_is_neg or not node_is_pos
        mispredicted_negative = (node_is_neg and score >=0)
        mispredicted_positive = (node_is_pos and score <=0)
        result = mispredicted_negative or mispredicted_positive
        if result:
            print("\n!!! WEIGHT UPDATE CONDITION !!!\n")


class SmallMarginOnCurrentNodeUpdateCondition(WeightUpdateCondition):
    """Update if the current node has small margin or was predicted incorrectly.

    """

    def should_update(self,
                      learning_state: LearningSearchState,
                      current_cost_state_pair: (float, sspace.SearchState),
                      next_cost_state_pairs,
                      feedback: MultiBinaryFeedback):
        cost, node = current_cost_state_pair
        score = -cost
        node_is_pos = node in feedback.positive_points
        node_is_neg = node in feedback.negative_points
        assert not node_is_neg or not node_is_pos
        small_margin_negative = (node_is_neg and score > -1)
        small_margin_positive = (node_is_pos and score < 1)
        result = small_margin_positive or small_margin_negative
        if result:
            print("\n!!! WEIGHT UPDATE CONDITION !!!\n")


class UpdateOnAnyFeedbackCondition(WeightUpdateCondition):
    """Update condition that specifies that update be made on every feedback.
    
    """
    def should_update(self,
                      learning_state: LearningSearchState,
                      current_cost_state_pair: (float, sspace.SearchState),
                      next_cost_state_pairs,
                      feedback: MultiBinaryFeedback):
        return True

class PreferenceFeedbackUpdateCondition(UpdateOnAnyFeedbackCondition):
    """Update the weight vector upon receiving any preference feedback.

    """
    pass

##------------------------------------------------------------------------------
## Update rules

class AdditiveUpdateRule(WeightUpdateRule, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute_delta(self, weigth_vector, vectors_add, vectors_subtract):
        return np.zeros_like(weight_vector)

    @abc.abstractmethod
    def vectors_to_add(self, feedback: UserFeedback):
        return []

    @abc.abstractmethod
    def vectors_to_subtract(self, feedback: UserFeedback):
        return []

    def update_weights(self, weight_vector,
                       feedback: UserFeedback,
                       learning_state: LearningSearchState):
        delta = self.compute_delta(weight_vector,
                                   self.vectors_to_add(feedback),
                                   self.vectors_to_subtract(feedback))
        return weight_vector + delta


class PerceptronAdditiveUpdateRule(AdditiveUpdateRule):
    def compute_delta(self, weigth_vector, positive_vectors, negative_vectors):
        delta = np.zeros_like(weigth_vector, float)
        if len(positive_vectors) != 0:
            pos_delta = np.average(positive_vectors, axis=0)
            delta += pos_delta
        if len(negative_vectors) != 0:
            neg_delta = np.average(negative_vectors, axis=0)
            delta -= neg_delta
        return delta


class AggressiveAdditiveUpdateRule(AdditiveUpdateRule):

    def __init__(self, margin=1.0):
        self.margin = float(margin)

    def _loss(self, w, x, y):
        epsilon = 1e-6
        return max(0, self.margin + epsilon - y * np.dot(w, x))

    def compute_delta(self, weight_vector, positive_vectors, negative_vectors):
        positive_vectors = [vec for vec in positive_vectors
                            if self._loss(weight_vector, vec, 1.0) > 0]
        negative_vectors = [vec for vec in negative_vectors
                            if self._loss(weight_vector, vec, -1.0) > 0]
        # print("Pos vectors:")
        # pprint.pprint(positive_vectors)
        # print("Neg vectors:")
        # pprint.pprint(negative_vectors)
        delta = np.zeros_like(weight_vector, float)
        if len(positive_vectors) != 0:
            pos_avg = np.average(positive_vectors, axis=0)
            norm_x_squared = np.linalg.norm(pos_avg) ** 2
            if norm_x_squared != 0:
                loss = self._loss(weight_vector, pos_avg, 1.0)
                delta += pos_avg * loss / norm_x_squared
            # print("Pos avg: {}".format(pos_avg))
            # print("Pos loss: {}".format(loss))
        if len(negative_vectors) != 0:
            neg_avg = np.average(negative_vectors, axis=0)
            norm_x_squared = np.linalg.norm(neg_avg) ** 2
            if norm_x_squared != 0:
                loss = self._loss(weight_vector, neg_avg, -1.0)
                delta -= neg_avg * loss / norm_x_squared
            # print("Neg avg: {}".format(neg_avg))
            # print("Neg loss: {}".format(loss))
        return delta


class BinaryAdditiveUpdateRule(AdditiveUpdateRule):
    def vectors_to_add(self, feedback: MultiBinaryFeedback):
        assert isinstance(feedback, MultiBinaryFeedback)
        return feedback.positive_points

    def vectors_to_subtract(self, feedback: MultiBinaryFeedback):
        assert isinstance(feedback, MultiBinaryFeedback)
        return feedback.negative_points


class PreferenceAdditiveUpdateRule(AdditiveUpdateRule):
    # TODO: Think again about the rules for preference updates, in particular
    #       if aggressive pairwise updates should go in batch, or one by one.
    #       Possibly, implement them based on feature vectors of the rankings,
    #       as per Joachims.

    def vectors_to_add(self, feedback: MultiSelectFeedback):
        assert isinstance(feedback, MultiSelectFeedback)
        return [(pref - nonpref)
                for pref in feedback.preferred
                for nonpref in feedback.other]

    def vectors_to_subtract(self, feedback: MultiSelectFeedback):
        assert isinstance(feedback, MultiSelectFeedback)
        return []


class PerceptronBinaryUpdateRule(BinaryAdditiveUpdateRule,
                                 PerceptronAdditiveUpdateRule):
    pass


class AggressiveBinaryUpdateRule(BinaryAdditiveUpdateRule,
                                 AggressiveAdditiveUpdateRule):
    """Implements the aggressive large-margin update requiring: y * w'x >= 1.

    """
    pass

class PerceptronPreferenceUpdateRule(PreferenceAdditiveUpdateRule,
                                     PerceptronAdditiveUpdateRule):
    """Implements the perceptron preference update: w <- w + x_best - x_other.

    """
    pass


class AggressivePreferenceUpdateRule(PreferenceAdditiveUpdateRule,
                                     AggressiveAdditiveUpdateRule):
    pass


##------------------------------------------------------------------------------
## Restart rules

class OnDequeingNegativeRestartFromScratchRule(RestartOnFeedbackRule):
    """Restarts from scratch when getting negative feedback on dequeued node.

    """
    def restart_states(self,
                       learning_state: LearningSearchState,
                       start_state: sspace.SearchState,
                       current_cost_state_pair: (float, sspace.SearchState),
                       next_cost_state_pairs,
                       feedback: MultiBinaryFeedback):
        cost, state = current_cost_state_pair
        if state in feedback.negative_points:
            return [start_state]
        else:
            return None


class OnPreferenceDisagreementRestartFromScratchRule(RestartOnFeedbackRule):
    """Restarts from scratch on any disagreement with given preference feedback.

    """

    def __init__(self, beta=0):
        self.beta = beta

    def restart_states(self,
                       learning_state: LearningSearchState,
                       start_state: sspace.SearchState,
                       current_cost_state_pair: (float, sspace.SearchState),
                       next_cost_state_pairs,
                       feedback: MultiSelectFeedback):
        scores = {state : -cost for cost, state in next_cost_state_pairs}
        pref_scores = [(scores[node], node) for node in feedback.preferred]
        other_scores = [(scores[node], node) for node in feedback.other]
        if len(pref_scores) == 0 or len(other_scores) == 0:
            # Either all nodes are preferred or all not preferred.
            return None
        key = operator.itemgetter(0)
        if min(pref_scores, key=key)[0] <\
                (1 - self.beta) * max(other_scores, key=key)[0]:
            pref_nodes_str = ["({:.4f}, {})".format(score, node.action())
                              for score, node in sorted(pref_scores, key=key)]
            other_nodes_str = ["({:.4f}, {})".format(score, node.action())
                              for score, node in sorted(other_scores, key=key)]
            print("Disagreement with preference feedback: "
                  "Preferred nodes: {} ; Nonpreferred nodes {} ; "
                  .format(pref_nodes_str, other_nodes_str))
            return [start_state]
        else:
            return None


##------------------------------------------------------------------------------
## Teachers

class BinaryFeedbackAlwaysPositiveTeacher(Teacher):
    """Dumb teacher for binary feedback that marks all given nodes as positive.
    
    Useful for profiling.
    """
    def get_feedback(self, query: MultiBinaryFeedback):
        assert isinstance(query, MultiBinaryFeedback)
        feedback = query.copy()
        for index, state in enumerate(query.points):
            feedback.labels[index] = MultiBinaryFeedback.POSITIVE_FEEDBACK
        return feedback


class BinaryFeedbackStdInTeacher(Teacher):
    """Teacher that asks for the binary user feedback from stdin.

    """
    def __init__(self, state_to_str):
        self._state_to_str = state_to_str

    def get_feedback(self, query: MultiBinaryFeedback):
        assert isinstance(query, MultiBinaryFeedback)
        feedback = query.copy()
        for index, state in enumerate(query.points):
            label = self.get_label_for_state(state)
            feedback.labels[index] = label
        return feedback

    def get_label_for_state(self, state):
        while True:
            print("Please, provide feedback for the following state:\n{}"
                  .format(self._state_to_str(state)))
            print(
                "Press 'y' if relevant, 'n' if irrelevant, 'x' if in doubt, "
                "and 'q' if you want to quit this.")
            answer = util.getch()
            if answer == 'y':
                return MultiBinaryFeedback.POSITIVE_FEEDBACK
            elif answer == 'n':
                return MultiBinaryFeedback.NEGATIVE_FEEDBACK
            elif answer == 'x':
                return MultiBinaryFeedback.NOT_SURE_FEEDBACK
            elif answer == 'q':
                raise SystemExit("Bye!")
            else:
                print(
                    "Sorry, '{}' is an invalid input. Please press 'y', 'n' "
                    "or 'x'."
                    .format(answer))


class PreferenceFeedbackStdInTeacher(Teacher):
    """Teacher that asks for the preference user feedback from stdin.

    """
    def __init__(self, state_to_str):
        self._state_to_str = state_to_str

    def get_feedback(self, query: MultiSelectFeedback):
        assert isinstance(query, MultiSelectFeedback)
        feedback = query.copy()
        all_nodes = feedback.other.copy()
        remaining_indices = list(range(len(all_nodes)))
        while len(remaining_indices) > 0:
            selected_indices = self._select_best_state(remaining_indices.copy(),
                                                       all_nodes)
            if selected_indices is None:
                break
            elif selected_indices == 'u':  # Undo
                feedback = query.copy()
                all_nodes = feedback.other.copy()
                remaining_indices = list(range(len(all_nodes)))
            else:
                for selected_idx in selected_indices:
                    selected_node = all_nodes[selected_idx]
                    feedback.preferred.append(selected_node)
                    feedback.other.remove(selected_node)
                    remaining_indices.remove(selected_idx)
        return feedback

    def _print_states(self, indices, states):
        """Prints the numbered list of states, each on a separate line.

        """
        result = ""
        # Note 1-based indexing.
        for index in indices:
            state = states[index]
            result += "{}. {}".format(index + 1, self._state_to_str(state))
            if index != indices[-1]:
                result += "\n"
        return result

    def _select_best_state(self, indices, states):
        assert len(states) <= 9
        digits = [str(i + 1) for i in indices]
        while True:
            print("Please, select some nodes that you are sure you want to add:\n{}"
                  .format(self._print_states(indices, states)))
            print(
                "Press the number of the node you wish to add, "
                "'n' if none, 'a' if all nodes are good, 'u' for undo, "
                "and 'q' if you want to quit this.")
            answer = util.getch()
            if answer in digits:
                # Convert from 1-based to array index.
                return [int(answer) - 1]
            if answer == 'a':  # All
                return indices.copy()
            elif answer in 'n' or 'd':
                return None
            elif answer == 'q':
                raise SystemExit("Bye!")
            elif answer in ['u']:
                return answer
            else:
                print(
                    "Sorry, '{}' is an invalid input. "
                    "Please press a digit in {}, 'u', 'a', 'n' or 'q'."
                    .format(answer, indices))


# class SessionCachingTeacher(Teacher):
#
#     def get_feedback(self, states: list[sspace.SearchState]) -> UserFeedback:
#         key = self._key(state)
#         feedback_value = self._cache.get(key)
#         if feedback_value is None:
#             feedback_value = self._delegate.get_feedback(state)
#             self._cache[key] = feedback_value
#         else:
#             pass
#         return self._cache[key]
