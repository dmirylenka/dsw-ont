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

import collections
import funktown
import logging
import operator

from dswont import dbpedia
from dswont import features as ftr
from dswont import learning_search as lsearch
from dswont import search
from dswont import search_space as sspace
from dswont import topics
from dswont import util
from dswont import wikiapi as wiki

logging.basicConfig(level=logging.WARN)

NODE_STATE_UNVIEWED = 0
NODE_STATE_CANDIDATE = 1
NODE_STATE_EXPLORED = 2

NODE_STATES = {
    NODE_STATE_UNVIEWED: 'unviewed',
    NODE_STATE_CANDIDATE: 'candidate',
    NODE_STATE_EXPLORED: 'explored'
}

class CategoryGraphState(sspace.State):
    """The state of the category graph.

    Includes the 'explored' nodes — those considered relevant according to the
    graph, and candidate nodes - those on the border of the relevant categories.

    """
    @classmethod
    def initial_state(cls, root):
        node_info = collections.OrderedDict()
        node_info[root] = {'status': NODE_STATE_CANDIDATE, 'depth_estimate': 0}
        return CategoryGraphState(funktown.ImmutableDict(node_info),
                                  size=0, hash_value=hash(root), last_node=root)

    def __init__(self, node_info, size, hash_value, last_node):
        if node_info is None:
            node_info = funktown.ImmutableDict()
        self._size = size
        self._node_info = node_info
        self._hash_value = hash_value
        self._last_node = last_node

    def size(self):
        return self._size

    def explored_nodes(self):
        return (node for node, info in self._node_info.items()
                if info['status'] == NODE_STATE_EXPLORED)

    def candidate_nodes(self):
        return (node for node, info in self._node_info.items()
                if info['status'] == NODE_STATE_CANDIDATE)

    def is_explored(self, node):
        node_info = self._node_info.get(node)
        return node_info is not None \
            and node_info['status'] == NODE_STATE_EXPLORED

    def is_candidate(self, node):
        node_info = self._node_info.get(node)
        return node_info is not None \
            and node_info['status'] == NODE_STATE_CANDIDATE

    def depth(self, node):
        node_info = self._node_info.get(node)
        return node_info['depth_estimate'] if node_info else None

    def predict(self, topics):
        return [self.is_explored(dbpedia.to_title(topic))
                for topic in topics]

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        if other is None:
            return False
        elif self is other:
            return True
        elif not isinstance(other, CategoryGraphState):
            return False
        elif self._hash_value != other._hash_value:
            return False
        else:
            return set(self.explored_nodes()) == set(other.explored_nodes())

    def __str__(self):
        return "{}+'{}'".format(self._size - 1, self._last_node)

    def __repr__(self):
        return "CategoryGraphState({})".format(repr(self._node_info))


def parent_on_the_shortest_known_way_to_root(node: str,
                                             state: CategoryGraphState,
                                             rel: wiki.CategoryRelationCache):
    parents = [p for p in rel.parents(node)
               if state.is_explored(p)]
    return min(parents, key=lambda p: state.depth(p)) if parents else None


def trace_back_to_node(node: str,
                       sstate: sspace.SearchState,
                       rel: wiki.CategoryRelationCache):
    while sstate and sstate.action():
        if sstate.action().node() == node:
            return sstate
        else:
            sstate = sstate.previous()
    return None


def node_parent_state(sstate: 'sspace.SearchState[CategoryGraphState]',
                      rel: wiki.CategoryRelationCache):
    """Returns the state that introduced the parent of the current state's node.

    """
    node = sstate.action().node()
    state = sstate.state()
    parent = parent_on_the_shortest_known_way_to_root(node, state, rel)
    result = trace_back_to_node(parent, sstate, rel)
    return result
    

# TODO: finish implementing this (discovering the first irrelevant node on the path from root)


class AddNodeAction(sspace.Action):
    """Action that describes moving between CategoryGraphState-s.

    The move represents adding one candidate node to the category graph.
    The current implementation of AddNodeAction and CategoryGraphState through
    persistent immutable dictionaries allows this operation in logarithmic time
    and logarithmic additional space.

    """
    def __init__(self, node, relation_cache: wiki.CategoryRelationCache):
        self._node = node
        self._relations = relation_cache

    def node(self):
        return self._node

    def next(self, state: CategoryGraphState):
        node_added = self._node
        assert state.is_candidate(node_added), \
            "Node '{}' must be candidate:".format(node_added)

        children = self._relations.children(node_added)
        parents = self._relations.parents(node_added)
        current_node_infos = state._node_info

        # Change the status of the node being added.
        node_info = current_node_infos.get(node_added)
        node_depth = node_info['depth_estimate']

        updated_node_infos = current_node_infos.assoc(
            node_added,
            {'status': NODE_STATE_EXPLORED,
             'depth_estimate': node_depth})

        # Change the status of the node's children.
        for child in children:
            child_node_info = (current_node_infos.get(child)
                               or {'status': NODE_STATE_UNVIEWED,
                                   'depth_estimate': float('inf')})
            # Note: makes use of the fact that NODE_STATE_UNVIEWED == 0
            child_status = child_node_info['status'] or NODE_STATE_CANDIDATE
            # Re-estimate the child's depth
            child_depth = child_node_info['depth_estimate']
            child_depth = min(node_depth + 1, child_depth)

            updated_node_infos = updated_node_infos.assoc(
                child,
                {'status': child_status, 'depth_estimate':child_depth})

        # Return the 'updated' state that structurally shares most of the data.
        hash_value = operator.xor(hash(state), hash(node_added))
        return CategoryGraphState(updated_node_infos,
                                  state.size() + 1, hash_value, node_added)

    def __hash__(self):
        return hash(self._node)

    def __eq__(self, other):
        if other is None:
            return False
        elif self is other:
            return True
        elif not isinstance(other, AddNodeAction):
            return False
        else:
            return self._node == other._node

    def __str__(self):
        return "+'{}'".format(self._node)

    def __repr__(self):
        return "AddNodeAction({})".format(self._node)


class CategoryGraphStateSpace(sspace.StateSpace):
    """The search space on CategoryGraphState-s.

    For a given category graph, the state defines the next possible states as
    those resulting from adding one candidate (adjacent) node to the graph.

    """
    def __init__(self, relations: wiki.CategoryRelationCache):
        self._relations = relations

    def next_actions(self, state: CategoryGraphState):
        return [AddNodeAction(candidate, self._relations)
                for candidate in state.candidate_nodes()]


##==============================================================================
## Definition of features
##==============================================================================

def relevant_links_incremental_fn(rel: wiki.CategoryRelationCache,
                                           normalized=True):
    def relevant_links_incremental(action: AddNodeAction,
                                   prev_sstate: sspace.SearchState,
                                   prev_value):
        new_links = set()
        state = prev_sstate.state()
        nnodes = state.size()
        # if nnodes == 0:
        #     return 1
        node = action.node()
        for parent in rel.parents(node):
            if state.is_explored(parent):
                new_links.add((parent, node))
        for child in rel.children(node):
            if state.is_explored(child):
                new_links.add((node, child))
        if normalized:
            return (nnodes * prev_value + len(new_links)) / (nnodes + 1)
        else:
            return prev_value + len(new_links)

    return relevant_links_incremental


def irrelevant_links_incremental_fn(rel: wiki.CategoryRelationCache,
                                             normalized=True):
    """Computes the average number of links from irrelevant to relevant nodes.

    The value is computed incrementally based on the new added node.
    
    """

    def irrelevant_links_incremental(action: AddNodeAction,
                                     prev_sstate: sspace.SearchState,
                                     prev_value):
        links_added = set()
        links_removed = set()
        state = prev_sstate.state()
        nnodes = state.size()
        node = action.node()
        for parent in rel.parents(node):
            if not state.is_explored(parent):
                links_added.add((parent, node))
        for child in rel.children(node):
            if state.is_explored(child):
                links_removed.add((node, child))
        nlink_diff = len(links_added) - len(links_removed)
        if normalized:
            return (nnodes * prev_value + nlink_diff) / (nnodes + 1)
        else:
            return prev_value + nlink_diff

    return irrelevant_links_incremental


def depth_incremental_fn(normalized=True):
    def depth_incremental(action, prev_sstate, prev_value):
        """Computes the average estimated depth of the explored nodes.
    
        The value is computed incrementally based on the depth of the added node.
    
        """
        state = prev_sstate.state()
        nnodes = state.size()
        node_depth = state.depth(action.node())
        if normalized:
            return (nnodes * prev_value + node_depth) / (nnodes + 1)
        else:
            return prev_value + node_depth
    return depth_incremental


def max_depth_incremental(action, prev_sstate, prev_value):
    """Computes the average estimated depth of the explored nodes.

    The value is computed incrementally based on the depth of the added node.

    """
    node_depth = prev_sstate.state().depth(action.node())
    if node_depth:
        return max(prev_value, node_depth)
    else:
        return 0


def graph_size_fn(normalized=True):
    def graph_size(sstate: sspace.SearchState):
        """Computes the inverse of the size of the category graph.
    
        """
    
        # Note that we are computing the size of the category graph indirectly,
        # through the previous state. This because our current state may be lazy,
        # and we want to avoid materializing it.
        previous_sstate = sstate.previous()
        if previous_sstate is None:
            return 0
        else:
            nnodes = previous_sstate.state().size() + 1
            if normalized:
                return 1 - 1.0 / nnodes
            else:
                return nnodes
    return graph_size


def leaf_nodes_incremental_fn(rel: wiki.CategoryRelationCache,
                              normalized=True):
    """Computes the average number of links from irrelevant to relevant nodes.

    The value is computed incrementally based on the new added node.
    
    """

    def leaf_nodes_incremental(action: AddNodeAction,
                                             prev_sstate: sspace.SearchState,
                                             prev_value):
        state = prev_sstate.state()
        nnodes = state.size()
        if nnodes == 0:
            return 0
        node = action.node()
        is_leaf = int(len(rel.children(node)) == 0)
        if normalized:
            return (nnodes * prev_value + is_leaf) / (nnodes + 1)
        else:
            return prev_value + is_leaf

    return leaf_nodes_incremental


def parent_child_similarity_incremental_fn(rel: wiki.CategoryRelationCache,
                                           normalized=True):
    """Computes the average similarity between all relevant parent-child pairs.

    The similarity is measures as the Jaccard index between the sets of
    non-stopword stems in parent's and child's titles.
    The value is computed incrementally based on the new added node.

    """

    # Sometimes, Wikipedia's sub- and super-category functionality is
    # inconsistent (because the lists of sub- and super-categories are cached
    # independently). Thus it may happen that a subcategory of A, will not see
    # A as its supercategory. As a result, some of the nodes that get into this
    # function may have no supercategories. We want to issue a warning about
    # such situations, but only the first time we see the problematic node.
    nodes_with_no_parents = {}

    def parent_child_similarity_incremental(action: AddNodeAction,
                                            prev_sstate: sspace.SearchState,
                                            prev_value):
        state = prev_sstate.state()
        nnodes = state.size()
        node = action.node()
        all_parents = (parent_node for parent_node in rel.parents(node))
        explored_parents = (parent_node for parent_node in all_parents
                            if state.is_explored(parent_node))
        similarity = max((util.word_jaccard(node, parent_node)
                          for parent_node in explored_parents),
                         default=None)
        if similarity is None:
            similarity = 0
            if node in nodes_with_no_parents:
                logging.warning("Explored node '{}' has no explored parents."
                                .format(node))
                nodes_with_no_parents.add(node)

        if normalized:
            return (nnodes * prev_value + similarity) / (nnodes + 1)
        else:
            return prev_value + similarity

    return parent_child_similarity_incremental


##==============================================================================
## Definition of the interactive learning procedure.
##==============================================================================


class HardcodedFeedbackCache(lsearch.FeedbackCache):

    def __init__(self):
        from dswont.learning_search import MultiBinaryFeedback as MBF
        self.node_labels = {
            'Computer science stubs' : MBF.POSITIVE_FEEDBACK,
            'Computer scientists' : MBF.POSITIVE_FEEDBACK,
            'Philosophy of computer science' : MBF.POSITIVE_FEEDBACK,
            'History of computer science' : MBF.POSITIVE_FEEDBACK,
            'Wikipedia books on computer science' : MBF.POSITIVE_FEEDBACK,
            'Computer science awards' : MBF.POSITIVE_FEEDBACK,
            'Computer science conferences' : MBF.POSITIVE_FEEDBACK,
            'Computer science organizations' : MBF.POSITIVE_FEEDBACK,
            'Unsolved problems in computer science' : MBF.POSITIVE_FEEDBACK,
            'Computer science portal' : MBF.POSITIVE_FEEDBACK,
            'Computer science literature' : MBF.POSITIVE_FEEDBACK,
            'Areas of computer science' : MBF.POSITIVE_FEEDBACK,
            'Computer science education' : MBF.POSITIVE_FEEDBACK
        }

    def restore_memoized_labels(self, feedback):
        updated_feedback = feedback.copy()
        if isinstance(updated_feedback, lsearch.MultiBinaryFeedback):
            knew_all_labels = True
            for index, state in enumerate(updated_feedback.points()):
                node = state.action().node()
                if node in self.node_labels:
                    print("Got the label from HARDCODED cache: {} : {}"
                          .format(node, self.node_labels[node]))
                    updated_feedback.labels[index] = self.node_labels[node]
                else:
                    knew_all_labels = False
            if knew_all_labels:
                # Count this feedback as user-given.
                updated_feedback.automatic = False
        else:
            raise TypeError("Unknown feedback type: {}".format(type(feedback)))
        return updated_feedback

    def memoize_labels(self, feedback):
        pass


class BinaryFeedbackParentCheckingTeacher(
        lsearch.FeedbackCollectionStepWrapper):

    def __init__(self,
                 delegate: lsearch.FeedbackCollectionStep,
                 rel: wiki.CategoryRelationCache):
        super().__init__(delegate)
        self._relations = rel

    def before(self,
               iteration: lsearch.LearningSearchIteration,
               log: lsearch.LearningSearchLog):
        return iteration

    def after(self,
              iteration: lsearch.LearningSearchIteration,
              log: lsearch.LearningSearchLog):
        if iteration.feedback_given:
            cost, parent_state = iteration.next_cost_state_pairs[0]
            parent_feedback = iteration.feedback_given
            feedback = parent_feedback
            all_feedback = [parent_feedback]
            while parent_state in parent_feedback.negative_points:
                state = parent_state
                feedback = parent_feedback
                parent_state = node_parent_state(state, self._relations)
                parent_feedback = self.delegate.get_feedback(
                    lsearch.MultiBinaryFeedback.query([parent_state]))
                all_feedback.append(parent_feedback)
            return iteration.update(
                feedback_given=lsearch.combine_multi_binary_feedbacks(
                    all_feedback))
        return iteration
                                    

def run_selection_procedure(max_nodes):
    with wiki.CategoryRelationCache() as rel:
        # root = 'Software'
        root = 'Computer science'
        start_state = CategoryGraphState.initial_state(root)

        state_space = CategoryGraphStateSpace(rel)

        relevant_links_ftr = ftr.CachingAdditiveSearchStateFeature(
            "relevant_links",
            relevant_links_incremental_fn(rel, False),
            zero_value=0)
        irrelevant_links_ftr = ftr.CachingAdditiveSearchStateFeature(
            "irrelevant_links",
            irrelevant_links_incremental_fn(rel, False),
            zero_value=0)
        leaves_ftr = ftr.CachingAdditiveSearchStateFeature(
            "frac_leaves",
            leaf_nodes_incremental_fn(rel, False),
            zero_value=0)
        parent_similarity_ftr = ftr.CachingAdditiveSearchStateFeature(
            "parent_similarity",
            parent_child_similarity_incremental_fn(rel, False),
            zero_value=1)
        graph_size_ftr = ftr.Feature('graph_size', graph_size_fn(False))


        depth_ftr = ftr.CachingAdditiveSearchStateFeature(
            'normalized_depth', depth_incremental_fn(False), zero_value=0)
        # max_depth_ftr = ftr.CachingAdditiveSearchStateFeature(
        #     'max_depth', max_depth_incremental, zero_value=0)
        features = ftr.Features(depth_ftr,
                                # # Max depth is a too aggressive feature.
                                # max_depth_ftr,
                                relevant_links_ftr,
                                irrelevant_links_ftr,
                                leaves_ftr,
                                parent_similarity_ftr,
                                graph_size_ftr
                               )
                               # # Exclude the product features for the moment.
                               #.add_products()
                               # # With preference updates, bias is never used.
                               #.add_bias()
        
        s0 = sspace.SearchState(start_state)
        planner = search.BeamSearchPlanner(1)
        
        def state_to_node_name(state):
            return "'{}'".format(state.action().node())

        def state_to_node_path(sstate):
            node_path = [sstate.action().node()]
            state = sstate.previous().state()
            parents = [p for p in rel.parents(node_path[-1])
                       if state.is_explored(p)]
            current_node = min(parents, key=lambda p: state.depth(p))
            while current_node is not None:
                node_path.append(current_node)
                parents = [p for p in rel.parents(node_path[-1])
                           if state.is_explored(p)]
                current_node = min(parents, key=lambda p: state.depth(p))\
                               if parents else None
            return ' -> '.join(reversed(node_path))

        # Weights for the ordinary features.
        # weight_vector = [-1, 1, -1, 1, 1]
        weight_vector = [0, 0, 0, 0, 0, 0]
        # Weights for the product features.
        # weight_vector += [0] * (features.n_features() - len(weight_vector))
        # # weight for the bias feature
        # weight_vector += [1]

        feedback_cache = lsearch.FeedbackCache()
        seen_feedback_filter = lsearch.OnlyAllowAGeneratedFeedbackPointOnce()
        # learner = lsearch.PassiveAggressivePreferenceLearner(weight_vector,
        #                                                      features, 1)
        learner = lsearch.SvmBasedPreferenceLearner(weight_vector, features)
        hardcoded_feedback = HardcodedFeedbackCache()

        in_memory_caching_feedback_teacher = lsearch.MemoizingFeedbackCollection(
            lsearch.BinaryFeedbackStdInTeacher(state_to_node_path),
            feedback_cache
        )

        hardcoded_feedback_teacher = lsearch.MemoizingFeedbackCollection(
            lsearch.BinaryFeedbackStdInTeacher(state_to_node_path),
            hardcoded_feedback
        )

        main_pipeline_feedback_teacher = BinaryFeedbackParentCheckingTeacher(
            in_memory_caching_feedback_teacher, rel)

        top_level_teacher = lsearch.MemoizingFeedbackCollection(
            hardcoded_feedback_teacher,
            feedback_cache)

        main_pipeline = lsearch.LearningSearchPipeline(
            (lsearch.NextNotMuchBetterThanCurrentQueryingCondition(0),
             lsearch.BinaryFeedbackOnNextNode(),
             main_pipeline_feedback_teacher,
             lsearch.PreferenceWrtPreviousFeedbackGeneration(),
             seen_feedback_filter,
             lsearch.AlwaysUpdateWeightsOnFeedbackCondition(),
             learner,
             lsearch.AlwaysRestartOnWeightUpdateCondition(),
             lsearch.NeverStopCondition()))

        top_level_pipeline = lsearch.LearningSearchPipeline(
            (lsearch.CurrentNodeHasSmallDepthQueryingCondition(0),
             lsearch.BinaryFeedbackOnAllNextNodes(),
             top_level_teacher,
             lsearch.PreferenceWrtPreviousFeedbackGeneration(),
             # lsearch.PairwisePreferenceFromBinaryFeedbackGeneration(),
             seen_feedback_filter,
             lsearch.AlwaysUpdateWeightsOnFeedbackCondition(),
             learner,
             lsearch.AlwaysRestartOnWeightUpdateCondition(),
             lsearch.NeverStopCondition()))
        
        periodic_feedback_pipeline = lsearch.LearningSearchPipeline(
            (lsearch.TooLongWithoutFeedbackQueryingCondition(200),
             lsearch.BinaryFeedbackOnNextNode(),
             main_pipeline_feedback_teacher,
             lsearch.PreferenceWrtPreviousFeedbackGeneration(),
             seen_feedback_filter,
             lsearch.PairwisePreferenceFromBinaryFeedbackGeneration(),
             lsearch.UpdateWeightsOnNegativeFeedbackCondition(),
             # lsearch.NextNotMuchBetterThanCurrentUpdateCondition(1),
             learner,
             # lsearch.AlwaysRestartOnWeightUpdateCondition(),
             lsearch.RestartOnNegativeFeedbackCondition(),
             lsearch.NeverStopCondition()))


        learning_algo = lsearch.LearningSearch(
                s0, state_space, planner, features, weight_vector,
                [top_level_pipeline, main_pipeline, periodic_feedback_pipeline])

        data = topics.default_data()
        accuracies = []
        weighted_f1s = []

        niter = 0
        while not learning_algo.done():
            cost, state = learning_algo.step()
            if niter % 10 == 0:
                accuracy = topics.evaluate_classifier(
                    state.state(), data.keys(),
                    data.values(), util.accuracy_score)
                weighted_f1 = topics.evaluate_classifier(
                    state.state(), data.keys(),
                    data.values(), util.weighted_f1)
                accuracies.append(accuracy)
                weighted_f1s.append(weighted_f1)
                node = state.action().node() if state.action() else "None"
                depth = state.state().depth(node)
                print("Evaluation at iteration {:>5}: accuracy {:4.3f},"
                      "weighted f1 {:4.3f}, topic: '{}', depth: {}, frontier {}"
                      .format(niter, accuracy, weighted_f1, node, depth, len(list(state.state().candidate_nodes()))))
            niter += 1
