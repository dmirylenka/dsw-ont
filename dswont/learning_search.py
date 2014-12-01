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
import sklearn

from dswont import features as ftr
from dswont import search
from dswont import search_space as sspace
from dswont import util


##==============================================================================
## Definition of the main data types used in the learning search algorithm.
##==============================================================================

class Feedback(metaclass=abc.ABCMeta):
    """'Marker interface' for feedback that can be given to the algorithm.

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

    @classmethod
    @abc.abstractmethod
    def points(cls):
        """Returns a query -- empty feedback without any user input.
        """
        return []


class LearningSearchLog(object):
    """The state of the learning search algorithm.

    This is the major data API between the components of the learning search.

    """

    def __init__(self, initial_weights):
        self._iteration = 0
        self._restarts  = []
        self._feedback  = util.DefaultOrderedDict(list)
        self._weights   = util.DefaultOrderedDict(list)
        self._weights[-1].append(initial_weights)

    def snapshot(self):
        return LearningSearchLogSnapshot(self, self._iteration)

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

    @property
    def last_weights(self):
        return self.weights.last_value[-1]

    def new_iteration(self):
        old_state = self.snapshot()
        self._iteration += 1
        return old_state

    def record_feedback(self, feedback: UserFeedbackInput):
        assert (not self._feedback) or max(self._feedback) <= self._iteration
        self._feedback[self._iteration].append(feedback)
        return self

    def record_restart(self):
        self._restarts.append(self._iteration)
        return self

    def record_update(self, weights):
        assert (not self._weights) or max(self._weights) <= self._iteration
        self._weights[self._iteration].append(weights)
        return self

    @property
    def n_restarts(self):
        return len(self.restarts)

    @property
    def moves_since_restart(self):
        last_restart = self.restarts[-1] if self.restarts else 0
        return self.iteration - last_restart

    @property
    def moves_since_feedback(self):
        last_feedback = self.feedback.last_key if self.feedback else 0
        return self.iteration - last_feedback

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
        return iteration <= self._iteration

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def restarts(self):
        return list(itertools.takewhile(self.valid_iter, self._log._restarts))

    @property
    def feedback(self):
        # return util.DefaultOrderedDict(
        #     itertools.takewhile(lambda item: self.valid_iter(item[0]),
        #                         self._log.feedback.items()))
        return self._log.feedback.take_while(
            lambda item: self.valid_iter(item[0]))

    @property
    def weights(self):
        # return util.DefaultOrderedDict(
        #     itertools.takewhile(lambda item: self.valid_iter(item[0]),
        #                         self._log.weights.items()))
        return self._log.weights.take_while(
            lambda item: self.valid_iter(item[0]))


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

def learning_search_iteration(current_cost_state_pair=None,
                              next_cost_state_pairs=None,
                              querying_condition=None, feedback_asked=None,
                              feedback_given=None, feedback_generated=None,
                              update_weights=None, weights=None, restart=None,
                              stop=None):
    if not feedback_generated:
        feedback_generated = []

    return LearningSearchIteration(current_cost_state_pair,
                                   next_cost_state_pairs,
                                   querying_condition, feedback_asked,
                                   feedback_given, feedback_generated,
                                   update_weights, weights, restart, stop)

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
            all_feedback_generated = iteration.feedback_generated.copy()
            feedback_generated = self.generate_feedback(iteration, log)
            if feedback_generated:
                all_feedback_generated += feedback_generated
                print("Generated feedback: {}".format(
                    util.format_list(feedback_generated)))
            return iteration.update(feedback_generated=all_feedback_generated)
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
            print("Update of norm {}"
                  .format(np.linalg.norm(new_weights-old_weights)))
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

class LearningSearch(search.FeatureBasedHeuristicSearch):
    """A search algorithm that learns from the feedback.

    """
    def __init__(self,
                 start: sspace.SearchState,
                 space: sspace.StateSpace,
                 planner: search.SearchPlanner,
                 features: ftr.Features,
                 initial_weight_vector,
                 iteration_pipelines,
                 **params):
        goal_test = lambda state: False
        super().__init__(start, space, planner, goal_test, features,
                         initial_weight_vector, **params)
        self._learning_log = LearningSearchLog(initial_weight_vector)
        self._pipelines = iteration_pipelines
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

        # print("\n### NEW STEP ###\n")

        current_cost, current_state = self._planner.peek()
        
        # print("Current state:")
        # pprint.pprint(list(map(str, current_state.action_seq())))

        learning_log = self._learning_log.new_iteration()
        restart = False
        first_pipeline = True

        for pipeline in self._pipelines:

            next_cost_state_pairs = self.next_cost_state_pairs()

            iteration = learning_search_iteration(
                current_cost_state_pair=(current_cost, current_state),
                next_cost_state_pairs=next_cost_state_pairs)

            if first_pipeline:
                self.report_iteration(iteration, learning_log)
                first_pipeline = False

            iteration = pipeline.process(iteration, learning_log)
        
            if iteration.feedback_given:
                self._learning_log.record_feedback(iteration.feedback_given)

            if iteration.update_weights:
                self.weight_vector = iteration.weights
                self._learning_log.record_update(iteration.weights)
            
            restart = restart or iteration.restart

        if restart:
            print("\n!!! RESTART !!!\n")
            self._learning_log.record_restart()
            self.restart([self._start])


        super().step()
        return current_cost, current_state


##==============================================================================
## Definition of some useful intermediate-level abstractions.
##==============================================================================

class FeedbackCollectionStepWrapper(FeedbackIterationStep):

    def __init__(self, delegate: FeedbackIterationStep):
        self.delegate = delegate

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        iteration = self.before(iteration, log)
        iteration = self.instead(iteration, log)
        iteration = self.after(iteration, log)
        return iteration

    @abc.abstractmethod
    def before(self,
               iteration: LearningSearchIteration,
               log: LearningSearchLog) -> LearningSearchIteration:
        return iteration

    def instead(self, iteration, log):
        return self.delegate.process(iteration, log)

    @abc.abstractmethod
    def after(self,
              iteration: LearningSearchIteration,
              log: LearningSearchLog) -> LearningSearchIteration:
        return iteration
        

class LearningSearchPipeline(list, FeedbackIterationStep):
    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        for step in self:
            iteration = step.process(iteration, log)
        return iteration


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

    def __hash__(self):
        return operator.xor(hash(self.item), hash(self.label))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, BinaryFeedback):
            return False
        else:
            return self.item == other.item and self.label == other.label


class PreferenceFeedback(Feedback,
                         collections.namedtuple('PreferenceFeedback',
                                                'preferred, other')):
    def map_features(self, features):
        return self._replace(preferred=features.compute_one(self.preferred),
                             other=features.compute_one(self.other))

    def __str__(self):
        return "({} > {})".format(str(self.preferred),
                                  str(self.other))

    def __hash__(self):
        return operator.xor(hash(self.preferred), hash(self.other))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, PreferenceFeedback):
            return False
        else:
            return self.preferred == other.preferred and\
                   self.other == other.other
    

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
        self._points = points
        self._labels = labels

    @classmethod
    def query(cls, points):
        empty_labels = [MultiBinaryFeedback.NO_FEEDBACK] * len(points)
        return cls(points, empty_labels)

    @property
    def positive_points(self):
        return [point for point, label in zip(self._points, self._labels)
                      if label == MultiBinaryFeedback.POSITIVE_FEEDBACK]

    @property
    def negative_points(self):
        return [point for point, label in zip(self._points, self._labels)
                      if label == MultiBinaryFeedback.NEGATIVE_FEEDBACK]

    @property
    def not_sure_points(self):
        return [point for point, label in zip(self._points, self._labels)
                      if label == MultiBinaryFeedback.NOT_SURE_FEEDBACK]

    def map_features(self, features: ftr.Features):
        features_feedback = self.copy()
        features_feedback._points = features.compute(self._points)
        return features_feedback

    def points(self):
        return self._points

    @property
    def labels(self):
        return self._labels

    def copy(self):
        return MultiBinaryFeedback(self._points.copy(),
                                   self._labels.copy())


class MultiSelectFeedback(UserFeedbackInput):
    """Feedback specifying selection of one set of points from a larger set.

    User selects the set of nodes A from a larger set B.
    It is assumed that points in the set A a preferred over the points in the
    set B\A. It is also assumed that the points in A are all positive, while the
    points in B\A may be positive or negative.

    """

    def __init__(self, preferred_points, other_points):
        self.preferred = list(preferred_points)
        self.other = list(other_points)

    @classmethod
    def query(cls, points):
        return cls([], list(points))

    def points(self):
        return self.preferred + self.other

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


class CurrentNodeHasSmallDepth(QueryingCondition,
                               collections.namedtuple(
                                   'CurrentNodeHasSmallDepth',
                                   ['current', 'depth'])):
    def __str__(self):
        return "CurrentNodeHasSmallDepth(current={}, depth={})"\
               .format(self.current, self.depth)


class TooLongWithoutFeedback(QueryingCondition,
                             collections.namedtuple(
                                 'CurrentNodeHasSmallDepth',
                                 ['current', 'next',
                                  'steps_without_feedback'])):
    def __str__(self):
        return "TooLongWithoutFeedback(current={}, next={}, "\
               "steps_without_feedback={})"\
               .format(self.current, self.next, self.steps_without_feedback)


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


class CurrentNodeHasSmallDepthQueryingCondition(QueryingConditionStep):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.states_seen = set()

    def querying_condition(self,
                  iteration: LearningSearchIteration,
                  log: LearningSearchLog):
        _, current_state = iteration.current_cost_state_pair
        depth = current_state.state().depth(current_state.action().node())
        if depth <= self.max_depth and current_state not in self.states_seen:
            self.states_seen.add(current_state)
            return CurrentNodeHasSmallDepth(current_state, depth)
        else:
            return None


class TooLongWithoutFeedbackQueryingCondition(QueryingConditionStep):

    def __init__(self, steps_without_feedback):
        self.steps_without_feedback = steps_without_feedback

    def querying_condition(self,
                  iteration: LearningSearchIteration,
                  log: LearningSearchLog):
        _, current_state = iteration.current_cost_state_pair
        depth = current_state.state().depth(current_state.action().node())
        if log.moves_since_feedback >= self.steps_without_feedback:
            _, next_state = iteration.next_cost_state_pairs[0]
            return TooLongWithoutFeedback(current_state, next_state,
                                          log.moves_since_feedback)
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


class BinaryFeedbackOnAllNextNodes(FeedbackTypeSelectionStep):

    def feedback_required(self,
                          iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> MultiBinaryFeedback:
        """Returns the type of feedback that has to be asked, or None.

        """
        next_states = [state for cost, state in iteration.next_cost_state_pairs]
        return MultiBinaryFeedback.query(next_states)


# TODO: introduce the selection step that only queries for the sub-categories of the current node?


##------------------------------------------------------------------------------
## Teachers

class FeedbackCache():

    def __init__(self):
        self.node_labels = {}

    def restore_memoized_labels(self, feedback):
        updated_feedback = feedback.copy()
        if isinstance(updated_feedback, MultiBinaryFeedback):
            for index, state in enumerate(updated_feedback.points()):
                node = state.action().node()
                if node in self.node_labels:
                    print("Got the label from cache: {} : {}"
                          .format(node, self.node_labels[node]))
                    updated_feedback.labels[index] = self.node_labels[node]
        elif isinstance(updated_feedback, MultiSelectFeedback):
            for state in updated_feedback.other:
                node = state.action().node()
                if node in self.node_labels:
                    label = self.node_labels[node]
                    if label == MultiBinaryFeedback.POSITIVE:
                        updated_feedback.positive_points.append(node)
                    elif label == MultiBinaryFeedback.NEGATIVE:
                        updated_feedback.negative_points.append(node)
                    elif label == MultiBinaryFeedback.NOT_SURE_FEEDBACK:
                        updated_feedback.not_sure_points.append(node)
                    else:
                        raise AssertionError("Shouldn't happen")
        else:
            raise TypeError("Unknown feedback type: {}".format(type(feedback)))
        return updated_feedback

    def memoize_labels(self, feedback):
        if isinstance(feedback, MultiBinaryFeedback):
            for state, label in zip(feedback.points(), feedback.labels):
                node = state.action().node()
                if label != MultiBinaryFeedback.NO_FEEDBACK:
                    self.node_labels[node] = label
        elif isinstance(feedback, MultiSelectFeedback):
            for state in feedback.preferred:
                node = state.action().node()
                self.node_labels[node] = MultiBinaryFeedback.POSITIVE
        else:
            raise TypeError("Unknown feedback type: {}".format(type(feedback)))
        return feedback


class MemoizingFeedbackCollection(FeedbackCollectionStepWrapper):

    def __init__(self, delegate: FeedbackCollectionStep,
                 feedback_cache: FeedbackCache):
        super().__init__(delegate)
        self.cache = feedback_cache
            
    def before(self,
               iteration: LearningSearchIteration,
               log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_asked:
            updated_feedback = self.cache.restore_memoized_labels(
                iteration.feedback_asked)
            return iteration.update(feedback_asked=updated_feedback)
        else:
            return iteration

    def instead(self, iteration, log):
        return self.delegate.process(iteration, log)

    def after(self,
              iteration: LearningSearchIteration,
              log: LearningSearchLog) -> LearningSearchIteration:
        if iteration.feedback_given:
            self.cache.memoize_labels(iteration.feedback_given)
        return iteration

class BinaryFeedbackAlwaysPositiveTeacher(FeedbackCollectionStep):
    """Dumb teacher for binary feedback that marks all given nodes as positive.
    
    Useful for profiling.
    """
    def get_feedback(self, query: MultiBinaryFeedback):
        assert isinstance(query, MultiBinaryFeedback)
        feedback = query.copy()
        for index, state in enumerate(query.points()):
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
        for index, state in enumerate(query.points()):
            if feedback._labels[index] == MultiBinaryFeedback.NO_FEEDBACK:
                label = self.get_label_for_state(state)
                feedback._labels[index] = label
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
        feedback = iteration.feedback_given
        assert isinstance(feedback, MultiBinaryFeedback)
        result = []
        for next_state in feedback.positive_points:
            result.append(PreferenceFeedback(next_state, current_state))
        for next_state in feedback.negative_points:
            result.append(PreferenceFeedback(current_state, next_state))
        return result if result else None


class PairwisePreferenceFromBinaryFeedbackGeneration(FeedbackGenerationStep):
    def generate_feedback(self, iteration: LearningSearchIteration,
                          log: LearningSearchLog) -> 'list[Feedback]':
        current_state = iteration.querying_condition.current
        feedback = iteration.feedback_given
        assert isinstance(feedback, MultiBinaryFeedback)
        result = []
        for pos_state in feedback.positive_points:
            for neg_state in feedback.negative_points:
                result.append(PreferenceFeedback(pos_state, neg_state))
        return result if result else None


class OnlyAllowAGeneratedFeedbackPointOnce(FeedbackIterationStep):

    def __init__(self):
        self.feedback_seen = set()

    def process(self,
                iteration: LearningSearchIteration,
                log: LearningSearchLog) -> LearningSearchIteration:
        new_feedback = []
        for feedback in iteration.feedback_generated:
            if feedback in self.feedback_seen:
                print("!!!!! Already seen feedback {}".format(feedback))
            else:
                new_feedback.append(feedback)
        self.feedback_seen.update(new_feedback)
        return iteration.update(feedback_generated=new_feedback)


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

    def __init__(self, initial_weights, features: ftr.Features, gamma=0):
        self._weights = initial_weights
        self.gamma = float(gamma)
        self.features = features

    def get_weights(self):
        return self._weights.copy()

    def update_weights(self, feedback: 'list[PreferenceFeedback]'):
        for preference in feedback:
            feature_preference = preference.map_features(self.features)
            preferred = feature_preference.preferred
            other = feature_preference.other
            increment = self.compute_increment(self._weights, preferred, other)
            print("Weights increment: {}".format(util.format_nums(increment, 4)))
            self._weights += increment

    def _loss(self, w, diff):
        epsilon = 1e-6  # To ensure the margin despite small numeric errors.
        return max(0, self.gamma + epsilon - np.dot(w, diff))

    def compute_increment(self, weight_vector, preferred, other):
        result = np.zeros_like(weight_vector, float)
        diff = preferred - other
        print("Current W: {}".format(weight_vector))
        print("Diff: {}".format(diff))
        loss = self._loss(weight_vector, diff)
        print("Loss: {}".format(loss))
        if loss > 0:
            norm_squared = np.linalg.norm(diff) ** 2
            if norm_squared != 0:
                result += diff * loss / norm_squared
        return result


class PerceptronPreferenceLearner(WeightUpdateStep):

    def __init__(self, initial_weights, features: ftr.Features):
        self._weights = initial_weights
        self.features = features

    def get_weights(self):
        return self._weights.copy()

    def update_weights(self, feedback: 'list[PreferenceFeedback]'):
        for preference in feedback:
            feature_preference = preference.map_features(self.features)
            preferred = feature_preference.preferred
            other = feature_preference.other
            increment = self.compute_increment(self._weights, preferred, other)
            print("Weights increment: {}".format(util.format_nums(increment, 4)))
            self._weights += increment

    def _loss(self, w, diff):
        return max(0, np.dot(w, diff))

    def compute_increment(self, weight_vector, preferred, other):
        return preferred - other


class SvmBasedPreferenceLearner(WeightUpdateStep):

    def __init__(self, initial_weights, features):
        self._weights = initial_weights
        self.features = features
        self.classifier = sklearn.svm.SVC(C=10.0, kernel='linear')
        self.data = []

    def get_weights(self):
        if hasattr(self.classifier, 'coef_'):
            # print("Weights: {}".format(
            #     np.array(*self.classifier.coef_, dtype=float)))
            return np.array(*self.classifier.coef_, dtype=float)
        else:
            return self._weights.copy()

    def update_weights(self, feedback: 'list[PreferenceFeedback]'):
        for preference in feedback:
            feature_preference = preference.map_features(self.features)
            preferred = feature_preference.preferred
            other = feature_preference.other
            diff = preferred - other
            print("Diff: {}".format(util.format_nums(diff, 4)))
            self.data.extend([(diff, 1), (-diff, -1)])
        X, y = zip(*self.data)
        self.classifier.fit(X, y)

##------------------------------------------------------------------------------
## Restart condition

class AlwaysRestartOnWeightUpdateCondition(RestartConditionStep):
    def should_restart(self,
                       iteration: LearningSearchIteration,
                       log: LearningSearchLog) -> bool:
        last_weights = log.last_weights
        possibly_changed_weights = iteration.weights
        print("Last weights: {}".format(util.format_nums(last_weights, 4)))
        if possibly_changed_weights is not None:
            print("Current weights: {}"
                  .format(util.format_nums(possibly_changed_weights, 4)))
            print("Close: {}"
                  .format(np.allclose(last_weights, possibly_changed_weights)))
        return (possibly_changed_weights is not None) and\
               not np.allclose(last_weights, possibly_changed_weights)

class NeverRestartOnWeightUpdateCondition(RestartConditionStep):
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
