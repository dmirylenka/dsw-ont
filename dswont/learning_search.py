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
import itertools
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

class Feedback(metaclass=abc.ABCMeta):
    """'Marker interface' for feedback that the can be given to the algorithm.

    """

    @abc.abstractmethod
    def map_features(self, features: ftr.Features):
        """Transforms all feedback points into their feature representation.

        Return the new object, leaving the current one unmodified.

        """
        pass


class UserFeedbackInput(metaclass=abc.ABCMeta):
    """Marker interface for the feedback that can be asked of the user.
    
    """
    @classmethod
    @abc.abstractmethod
    def query(cls, points):
        """Returns a query -- empty feedback without any user input.
        """
        return cls(points)


class LearningSearchLog(object):
    """The state of the learning search algorithm.

    This is the major data API between the components of the learning search.

    """

    def __init__(self, iteration=0, restarts=[], feedback=None, weights=None):
        self._iteration = iteration
        self._restarts = restarts
        self._feedback = feedback or collections.OrderedDict()
        self._weights = weights or collections.OrderedDict()

    def snapshot(self):
        return LearningSearchLogSnapshot(self, self.iteration)

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
        old_state = self.snapshot()
        self._iteration += 1
        return old_state

    def record_feedback(self, feedback: UserFeedbackInput):
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
        return len(self.restarts)

    @property
    def moves_since_restart(self):
        last_restart = self.restarts[-1] if self.restarts else 0
        return self.iteration - last_restart

    @property
    def n_feedback(self):
        return len(self.feedback)

    @property
    def n_updates(self):
        return len(self.weights)


class LearningSearchLogSnapshot(LearningSearchLog):
    
    def __init__(self, log, iteration):
        assert iteration <= log.iteration
        self._log = log
        self._iteration = iteration

    def valid_iter(self, iteration):
        return iteration < self._iteration

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def restarts(self):
        return list(itertools.takewhile(self.valid_iter, self._log._restarts))

    @property
    def feedback(self):
        return collections.OrderedDict(
            itertools.takewhile(lambda item: self.valid_iter(item[0]),
                                self._log.feedback.items()))

    @property
    def weights(self):
        return collections.OrderedDict(
            itertools.takewhile(lambda item: self.valid_iter(item[0]),
                                self._log.weights.items()))

    def snapshot(self, iteration):
        return LearningSearchLogSnapshot(self._log, iteration)


def disallow_modification(self, *args, **kwargs):
    raise ValueError('Cannot modify LearningSearchLogSnapshot')
LearningSearchLogSnapshot.record_feedback = disallow_modification
LearningSearchLogSnapshot.record_restart  = disallow_modification
LearningSearchLogSnapshot.record_update   = disallow_modification


class LearningSearchIteration(collections.namedtuple('LearningSearchIteration',
                                                    ['current_cost_state_pair',
                                                     'next_cost_state_pairs',
                                                     'querying_condition',
                                                     'feedback_asked',
                                                     'feedback_given',
                                                     'feedback_generated',
                                                     'update_weights',
                                                     'weights',
                                                     'restart',
                                                     'stop'])):
    def update(self, **kwargs):
        return self._replace(**kwargs)

# Allow for partial initialization of LearningSearchIteration.
LearningSearchIteration.__new__.__defaults__ = (None, None, None, None,
                                                None, None, None, None,
                                                None, None)

##==============================================================================
## Definition of the abstract components of the learning search algorithm.
##==============================================================================


class FeedbackIterationStep(metaclass=abc.ABCMeta):
    """A step in the pipeline of executing one iteration of the learning search.

    """
    @abc.abstractmethod
    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        return iteration


class QueryingCondition(metaclass=abc.ABCMeta):
    """Data about the contition in which we need to query the user.

    """
    pass
    

class QueryingConditionStep(FeedbackIterationStep):
    """Decides if the feedback should be asked of the user.

    """

    @abc.abstractmethod
    def querying_condition(self,
                           iteration: LearningSearchIteration,
                           log: LearningSearchLog) -> QueryingCondition:
        """Returns the data describing why feedback should be asked, or None.

        """
        pass

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        condition = self.querying_condition(iteration, log)
        if condition:
            print("Querying condition: {}".format(str(condition)))
        return iteration.update(querying_condition=condition)


class FeedbackTypeSelectionStep(FeedbackIterationStep):

    @abc.abstractmethod
    def feedback_required(self,
                          iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> UserFeedbackInput:
        """Returns the type of feedback that has to be asked, or None.

        """
        pass

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.querying_condition:
            feedback_asked = self.feedback_required(iteration, log)
            return iteration.update(feedback_asked=feedback_asked)
        else:
            return iteration


class Teacher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_feedback(self, query: UserFeedbackInput) -> UserFeedbackInput:
        pass


class FeedbackCollectionStep(FeedbackIterationStep, Teacher):

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_asked:
            # TODO: check if copying the feedback object is really needed here.
            print("Iteration {} since restart" .format(log.moves_since_restart))
            feedback_given = self.get_feedback(iteration.feedback_asked.copy())
            return iteration.update(feedback_given=feedback_given)
        else:
            return iteration


class FeedbackGenerationStep(FeedbackIterationStep):
    """Generating the feedback to the learner from the user feedback input.

    """

    @abc.abstractmethod
    def generate_feedback(self, iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> 'list[Feedback]':
        return []

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_given:
            feedback_generated = self.generate_feedback(iteration, log)
            if feedback_generated:
                print("Generated feedback: {}".format(
                    util.format_list(feedback_generated)))
            return iteration.update(feedback_generated=feedback_generated)
        else:
            return iteration


class WeightUpdateConditionStep(FeedbackIterationStep):
    """Decides if the weights should be updated.

    """
    @abc.abstractmethod
    def should_update_weights(self,
                              iteration: LearningSearchIteration,
                              log: LearningSearchLog) -> bool:
        return False

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_generated:
            update_weights = self.should_update_weights(iteration, log)
            print("Should update weights: {}".format(update_weights))
            return iteration.update(update_weights=update_weights)
        else:
            return iteration


class Learner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def update_weights(self, feedback: 'list[Feedback]'):
        pass


class WeightUpdateStep(FeedbackIterationStep, Learner):
    """Updates the weights.

    """
    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.update_weights:
            old_weights = self.get_weights().copy()
            self.update_weights(iteration.feedback_generated)
            new_weights = self.get_weights()
            print("Updated vector:\n{} +\n{} ->\n{}"
                  .format(util.format_nums(old_weights, 5),
                          util.format_nums(new_weights -
                                           old_weights, 10),
                          util.format_nums(new_weights, 5)))
            print("Iteration {} since restart, update of norm {}"
                  .format(log.moves_since_restart,
                          np.linalg.norm(new_weights-old_weights)))
            return iteration.update(weights=new_weights)
        else:
            return iteration


class RestartConditionStep(metaclass=abc.ABCMeta):
    """Decides if the search algorithm should be restarted (e.g. on mistake).

    Also decides which states the algorithm should restart from.
    
    """
    @abc.abstractmethod
    def should_restart(self,
                       iteration: LearningSearchIteration,
                       log: LearningSearchLog) -> bool:
        return False

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_generated:
            restart = self.should_restart(iteration, log)
            return iteration.update(restart=restart)
        else:
            return iteration


class StopConditionStep(metaclass=abc.ABCMeta):
    """Decides if the search algorithm should stop.

    """
    @abc.abstractmethod
    def should_stop(self,
                    iteration: LearningSearchIteration,
                    log: LearningSearchLog) -> bool:
        return False

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        stop = self.should_stop(iteration, log)
        return iteration.update(stop=stop)


##==============================================================================
## Definition of the learning search algorithm itself.
##==============================================================================

class LearningSearchPipeline(list, FeedbackIterationStep):
    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        for step in self:
            iteration = step.process(iteration, log)
        return iteration


class LearningSearch(search.FeatureBasedHeuristicSearch):
    """A search algorithm that learns from the feedback.

    """
    def __init__(self,
                 start: sspace.SearchState,
                 space: sspace.StateSpace,
                 planner: search.SearchPlanner,
                 features: ftr.Features,
                 initial_weight_vector,
                 iteration_pipeline,
                 **params):
        goal_test = lambda state: False
        super().__init__(start, space, planner, goal_test, features,
                         initial_weight_vector, **params)
        self._learning_log = LearningSearchLog()
        self._pipeline = iteration_pipeline
        super().step()

    def report_iteration(self,
                         iteration: LearningSearchIteration,
                         learning_log: LearningSearchLog):
        print(
            "Iteration {}, {} restarts, {} moves since restart, "
            "{} feedback points, {} updates, current vector: {}."
            .format(learning_log.iteration,
                    learning_log.n_restarts,
                    learning_log.moves_since_restart,
                    learning_log.n_feedback,
                    learning_log.n_updates,
                    util.format_nums(self._weight_vector, 5)))
        current_cost, current_state = iteration.current_cost_state_pair
        current_node = current_state.action().node()
        node_depth = current_state.state().depth(current_node)
        print(
            "Current state: {}(depth {}), {} nodes, "
            "features: {}, score: {}."
            .format(current_node,
                    node_depth,
                    current_state.state().size(),
                    list(self._features.value_map(current_state).items()),
                    -current_cost))


    def step(self) -> (float, sspace.SearchState):
        current_cost, current_state = self._planner.peek()
        
        # print("Current state:")
        # pprint.pprint(list(map(str, current_state.action_seq())))

        learning_log = self._learning_log.new_iteration()
        next_cost_state_pairs = self.next_cost_state_pairs()
        pipeline = self._pipeline

        iteration = LearningSearchIteration(
            current_cost_state_pair=(current_cost, current_state),
            next_cost_state_pairs=next_cost_state_pairs)

        self.report_iteration(iteration,
                              learning_log)

        iteration = pipeline.process(iteration, learning_log)
        
        if iteration.feedback_given:
            self._learning_log.record_feedback(iteration.feedback_given)

        if iteration.update_weights:
            self.weight_vector = iteration.weights
            self._learning_log.record_update(iteration.weights)

        if iteration.restart:
            print("\n!!! RESTART CONDITION !!!\n")
            self.restart([self._start])
            self._learning_log.record_restart()

        super().step()
        return current_cost, current_state


##==============================================================================
## Definition of the specific components of the learning search algorithm.
##==============================================================================

##------------------------------------------------------------------------------
## Feedback types

class BinaryFeedback(Feedback,
                     collections.namedtuple('BinaryFeedback', 'item, label')):

    POSITIVE = 1
    NEGATIVE = -1

    def map_features(self, features):
        return self._replace(item=features.compute_one(self.item))


class PreferenceFeedback(Feedback,
                         collections.namedtuple('PreferenceFeedback',
                                                'preferred, other')):
    def map_features(self, features):
        return self._replace(preferred=features.compute_one(self.preferred),
                             other=features.compute_one(self.other))

    def __str__(self):
        return "PreferenceFeedback({}>{})".format(str(self.preferred),
                                                  str(self.other))

##------------------------------------------------------------------------------
## User feedback input types

class MultiBinaryFeedback(UserFeedbackInput):
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

    @property
    def not_sure_points(self):
        return [point for point, label in zip(self.points, self.labels)
                      if label == MultiBinaryFeedback.NOT_SURE_FEEDBACK]

    def map_features(self, features: ftr.Features):
        features_feedback = self.copy()
        features_feedback.points = features.compute(self.points)
        return features_feedback

    def copy(self):
        return MultiBinaryFeedback(self.points.copy(),
                                   self.labels.copy())


class MultiSelectFeedback(UserFeedbackInput):
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


##------------------------------------------------------------------------------
## Querying conditions


class SmallAbsoluteGradient(QueryingCondition,
                            collections.namedtuple(
                                'SmallAbsoluteGradientCondition',
                                ['current', 'current_score',
                                 'next', 'next_score'])):
    pass


class NextNotMuchBetterThanCurrent(QueryingCondition,
                                   collections.namedtuple(
                                       'SmallAbsoluteGradientCondition',
                                       ['current', 'current_score',
                                       'next', 'next_score'])):
    def __str__(self):
        return "NextNotMuchBetterThanCurrent(current={}({}), next={}({}))"\
               .format(self.current, self.current_score,
                       self.next, self.next_score)


class SmallAbsoluteGradientQueryingCondition(QueryingConditionStep):

    def __init__(self, gamma):
        self.gamma = gamma

    def querying_condition(self,
                  iteration: LearningSearchIteration,
                  log: LearningSearchLog):
        current_cost, current_state = iteration.current_cost_state_pair
        best_next_cost, best_next_state = iteration.next_cost_state_pairs[0]
        if abs(-best_next_cost + current_cost) < self.gamma:
            return SmallAbsoluteGradient(current_state, -current_cost,
                                         best_next_state, -best_next_cost)
        else:
            return None


class NextNotMuchBetterThanCurrentQueryingCondition(QueryingConditionStep):

    def __init__(self, gamma):
        self.gamma = gamma

    def querying_condition(self,
                  iteration: LearningSearchIteration,
                  log: LearningSearchLog):
        current_cost, current_state = iteration.current_cost_state_pair
        best_next_cost, best_next_state = iteration.next_cost_state_pairs[0]
        if -best_next_cost + current_cost < self.gamma:
            return NextNotMuchBetterThanCurrent(
                current_state, -current_cost, best_next_state, -best_next_cost)
        else:
            return None
            

##------------------------------------------------------------------------------
## Feedback type selection

class BinaryFeedbackOnNextNode(FeedbackTypeSelectionStep):

    def feedback_required(self,
                          iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> MultiBinaryFeedback:
        """Returns the type of feedback that has to be asked, or None.

        """
        best_next_cost, best_next_state = iteration.next_cost_state_pairs[0]
        return MultiBinaryFeedback.query([best_next_state])


##------------------------------------------------------------------------------
## Teachers

class BinaryFeedbackAlwaysPositiveTeacher(FeedbackCollectionStep):
    """Dumb teacher for binary feedback that marks all given nodes as positive.
    
    Useful for profiling.
    """
    def get_feedback(self, query: MultiBinaryFeedback):
        assert isinstance(query, MultiBinaryFeedback)
        feedback = query.copy()
        for index, state in enumerate(query.points):
            feedback.labels[index] = MultiBinaryFeedback.POSITIVE_FEEDBACK
        return feedback


class BinaryFeedbackStdInTeacher(FeedbackCollectionStep):
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


class PreferenceFeedbackStdInTeacher(FeedbackCollectionStep):
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


##------------------------------------------------------------------------------
## Feedback generation from the user input

class PreferenceWrtCurrentFeedbackGeneration(FeedbackGenerationStep):
    def generate_feedback(self, iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> 'list[Feedback]':
        current_state = iteration.querying_condition.current
        next_state = iteration.querying_condition.next
        feedback = iteration.feedback_given
        assert isinstance(feedback, MultiBinaryFeedback)
        assert (len(feedback.positive_points) + len(feedback.negative_points)\
                + len(feedback.not_sure_points)) == 1
        if next_state in feedback.positive_points:
            return [PreferenceFeedback(next_state, current_state)]
        elif next_state in feedback.negative_points:
            return [PreferenceFeedback(current_state, next_state)]
        else:  # Not sure about the next state.
            return None

##------------------------------------------------------------------------------
## Update conditions

class AlwaysUpdateWeightsOnFeedbackCondition(WeightUpdateConditionStep):
    def should_update_weights(self,
                              iteration: LearningSearchIteration,
                              log: LearningSearchLog) -> bool:
        return True


##------------------------------------------------------------------------------
## Weight updates

class PassiveAggressivePreferenceLearner(WeightUpdateStep):

    def __init__(self, initial_weights, features, gamma=0):
        self.weights = initial_weights
        self.gamma = float(gamma)
        self.features = features

    def get_weights(self):
        return self.weights

    def update_weights(self, feedback: 'list[PreferenceFeedback]'):
        for preference in feedback:
            feature_preference = preference.map_features(self.features)
            preferred = feature_preference.preferred
            other = feature_preference.other
            increment = self.compute_increment(self.weights, preferred, other)
            self.weights += increment

    def _loss(self, w, diff):
        epsilon = 1e-6  # To ensure the margin despite small numeric errors.
        return max(0, self.gamma + epsilon - np.dot(w, diff))

    def compute_increment(self, weight_vector, preferred, other):
        result = np.zeros_like(weight_vector, float)
        diff = preferred - other
        # print("Diff: {}".format(diff))
        loss = self._loss(weight_vector, diff)
        # print("Loss: {}".format(loss))
        if loss > 0:
            norm_squared = np.linalg.norm(diff) ** 2
            if norm_squared != 0:
                result += diff * loss / norm_squared
        return result


class PerceptronPreferenceLearner(WeightUpdateStep):

    def __init__(self, initial_weights, features):
        self.weights = initial_weights
        self.features = features

    def get_weights(self):
        return self.weights

    def update_weights(self, feedback: 'list[PreferenceFeedback]'):
        for preference in feedback:
            feature_preference = preference.map_features(self.features)
            preferred = feature_preference.preferred
            other = feature_preference.other
            increment = self.compute_increment(self.weights, preferred, other)
            self.weights += increment

    def _loss(self, w, diff):
        return max(0, np.dot(w, diff))

    def compute_increment(self, weight_vector, preferred, other):
        return preferred - other

##------------------------------------------------------------------------------
## Restart condition

class AlwaysRestartOnFeedbackCondition(RestartConditionStep):
    def should_restart(self,
                       iteration: LearningSearchIteration,
                       log: LearningSearchLog) -> bool:
        return True

class NeverRestartOnFeedbackCondition(RestartConditionStep):
    def should_restart(self,
                       iteration: LearningSearchIteration,
                       log: LearningSearchLog) -> bool:
        return False

##------------------------------------------------------------------------------
## Stop condition

class NeverStopCondition(StopConditionStep):
    def should_stop(self,
                    iteration: LearningSearchIteration,
                    log: LearningSearchLog) -> bool:
        return False
