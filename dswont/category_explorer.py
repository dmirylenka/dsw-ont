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
        return CategoryGraphState(funktown.ImmutableDict(node_info), size=0)

    def __init__(self, node_info=None, size=0):
        if node_info is None:
            node_info = funktown.ImmutableDict()
        self._size = size
        self._node_info = node_info

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

    def __repr__(self):
        return "CategoryGraphState({})".format(repr(self._node_info))


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
        node_info = current_node_infos.get(node_added).copy()
        node_info['status'] = NODE_STATE_EXPLORED
        updated_node_infos = current_node_infos.assoc(node_added, node_info)

        node_depth = node_info.get('depth_estimate')

        # Change the status of the node's children.
        for child in children:
            child_node_info = (current_node_infos.get(child) or {}).copy()
            if child_node_info.get('status') != NODE_STATE_EXPLORED:
                child_node_info['status'] = NODE_STATE_CANDIDATE
            # Re-estimate the child's depth
            old_child_depth = child_node_info.get('depth_estimate') \
                or float('inf')
            child_node_info['depth_estimate'] = \
                min(node_depth + 1, old_child_depth)

            updated_node_infos = updated_node_infos.assoc(child,
                                                          child_node_info)

        # Return the 'updated' state that structurally shares most of the data.
        return CategoryGraphState(updated_node_infos, state.size() + 1)

    def __repr__(self):
        return "AddNodeAction('{}')".format(self._node)


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


################################################################################
## Definition of features
################################################################################

def relevant_links_per_node_incremental_fn(rel: wiki.CategoryRelationCache):
    """Computes corrected average number of links between the explored nodes.
    
    The precise value is (nlinks + 1) / nnodes.
    The value is computed incrementally based on the additional links.
    
    """

    def relevant_links_per_node_incremental(action: AddNodeAction,
                                            prev_sstate: sspace.SearchState,
                                            prev_value):
        new_links = set()
        state = prev_sstate.state()
        nnodes = state.size()
        if nnodes == 0:
            return 1
        node = action.node()
        for parent in rel.parents(node):
            if state.is_explored(parent):
                new_links.add((parent, node))
        for child in rel.children(node):
            if state.is_explored(child):
                new_links.add((node, child))
        return (nnodes * prev_value + len(new_links)) / (nnodes + 1)

    return relevant_links_per_node_incremental


def irrelevant_links_per_node_incremental_fn(rel: wiki.CategoryRelationCache):
    """Computes the average number of links from irrelevant to relevant nodes.

    The value is computed incrementally based on the new added node.
    
    """

    def irrelevant_links_per_node_incremental(action: AddNodeAction,
                                              prev_sstate: sspace.SearchState,
                                              prev_value):
        links_added = set()
        links_removed = set()
        state = prev_sstate.state()
        nnodes = state.size()
        if nnodes == 0:
            return 0
        node = action.node()
        for parent in rel.parents(node):
            if not state.is_explored(parent):
                links_added.add((parent, node))
        for child in rel.children(node):
            if state.is_explored(child):
                links_removed.add((node, child))
        nlink_diff = len(links_added) - len(links_removed)
        return (nnodes * prev_value + nlink_diff) / (nnodes + 1)

    return irrelevant_links_per_node_incremental


def depth_per_node_incremental(action, prev_sstate, prev_value):
    """Computes the average estimated depth of the explored nodes.
    
    The value is computed incrementally based on the depth of the added node.
    
    """
    state = prev_sstate.state()
    nnodes = state.size()
    node_depth = state.depth(action.node())
    return (nnodes * prev_value + node_depth) / (nnodes + 1)


def max_depth_incremental(action, prev_sstate, prev_value):
    """Computes the average estimated depth of the explored nodes.

    The value is computed incrementally based on the depth of the added node.

    """
    node_depth = prev_sstate.state().depth(action.node())
    if node_depth:
        return max(prev_value, node_depth)
    else:
        return 0


def normalized_graph_size(sstate: sspace.SearchState):
    """Computes the inverse of the size of the category graph.

    """

    # Note that we are computing the size of the category graph indirectly,
    # through the previous state. This because our current state may be lazy,
    # and we want to avoid materializing it.
    previous_sstate = sstate.previous()
    if previous_sstate is None:
        return 1.0
    else:
        return 1 - 1.0 / (previous_sstate.state().size() + 1)


def proportion_of_leaf_nodes_incremental_fn(rel: wiki.CategoryRelationCache):
    """Computes the average number of links from irrelevant to relevant nodes.

    The value is computed incrementally based on the new added node.
    
    """

    def proportion_of_leaf_nodes_incremental(action: AddNodeAction,
                                             prev_sstate: sspace.SearchState,
                                             prev_value):
        state = prev_sstate.state()
        nnodes = state.size()
        if nnodes == 0:
            return 0
        node = action.node()
        is_leaf = int(len(rel.children(node)) == 0)
        return (nnodes * prev_value + is_leaf) / (nnodes + 1)

    return proportion_of_leaf_nodes_incremental


def parent_child_similarity_incremental_fn(rel: wiki.CategoryRelationCache):
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
        if nnodes == 0:
            return 0
        node = action.node()
        all_parents = (parent_node for parent_node in rel.parents(node))
        explored_parents = (parent_node for parent_node in all_parents
                            if state.is_explored(parent_node))
        similarity = max((util.stem_jaccard(node, parent_node)
                          for parent_node in explored_parents),
                         default=None)
        if similarity is None:
            similarity = 0
            if node in nodes_with_no_parents:
                logging.warning("Explored node '{}' has no explored parents."
                                .format(node))
                nodes_with_no_parents.add(node)

        return (nnodes * prev_value + similarity) / (nnodes + 1)

    return parent_child_similarity_incremental


################################################################################
## Definition of the interactive learning procedure.
################################################################################

def run_selection_procedure(max_nodes):
    # new_subcat_index = '/Users/dmirylenka/data/dswont-fresh/uri-to-subcats'
    # new_supercat_index = '/Users/dmirylenka/data/dswont-fresh/uri-to-supercats'
    # with wiki.CategoryRelationCache(
    #         subcat_index_file=new_subcat_index,
    #         supercat_index_file=new_supercat_index) as rel:
    with wiki.CategoryRelationCache() as rel:
        root = 'Computing'
        start_state = CategoryGraphState.initial_state(root)

        state_space = CategoryGraphStateSpace(rel)

        relevant_links_ftr = ftr.CachingAdditiveSearchStateFeature(
            "relevant_links",
            relevant_links_per_node_incremental_fn(rel),
            zero_value=1)
        irrelevant_links_ftr = ftr.CachingAdditiveSearchStateFeature(
            "irrelevant_links",
            irrelevant_links_per_node_incremental_fn(rel),
            zero_value=0)
        leaves_ftr = ftr.CachingAdditiveSearchStateFeature(
            "frac_leaves",
            proportion_of_leaf_nodes_incremental_fn(rel),
            zero_value=0)
        parent_similarity_ftr = ftr.CachingAdditiveSearchStateFeature(
            "parent_similarity",
            parent_child_similarity_incremental_fn(rel),
            zero_value=0)
        graph_size_ftr = ftr.Feature('graph_size', normalized_graph_size)


        depth_ftr = ftr.CachingAdditiveSearchStateFeature(
            'normalized_depth', depth_per_node_incremental, zero_value=0)
        max_depth_ftr = ftr.CachingAdditiveSearchStateFeature(
            'max_depth', max_depth_incremental, zero_value=0)
        constant_feature = ftr.Feature('unity', lambda x: 1)
        features = ftr.Features(constant_feature,
                                depth_ftr,
                                # max_depth_ftr,
                                relevant_links_ftr,
                                irrelevant_links_ftr,
                                leaves_ftr,
                                parent_similarity_ftr,
                                graph_size_ftr)

        def goal_test(sstate):
            return sstate.state().size() >= max_nodes

        s0 = sspace.SearchState(start_state)
        planner = search.BeamSearchPlanner(1)
        update_rule = lsearch.AggressiveUpdateRule()
        # update_rule = PerceptronUpdateRule()
        restart_rule = lsearch.RestartFromScratchRule()
        search_learner = lsearch.LearningSearch(s0, state_space, planner,
                                                goal_test, features,
                                                update_rule, restart_rule,
                                                [1, -1, 1, -1, 1, 1, -1])

        def state_to_node_name(state):
            return "'{}'".format(state.action().node())

        def state_to_node_pair(state):
            previous_action = state.previous().action()
            previous_node = previous_action.node() if previous_action else None
            return "'{}'->'{}'".format(previous_node, state.action().node())

        data = topics.default_data()

        def ground_truth_feedback_fn(state):
            node_name = dbpedia.to_category_uri(state.action().node())
            label = data.get(node_name)
            if label is None:
                return '0'
            elif label:
                return '+'
            else:
                return '-'

        teacher = lsearch.SessionCachingTeacher(
            lsearch.StdInUserFeedbackTeacher(state_to_str=state_to_node_pair),
            # AbstractTeacher(ground_truth_feedback_fn),
            key=state_to_node_name)

        learning_algo = lsearch.LearningSearchAlgorithm(
            search_learner, teacher,
            state_to_str=state_to_node_pair,
            alpha=0.2,
            steps_no_feedback=max_nodes)

        n_explored = []
        n_candidate = []
        n_total = []
        accuracies = []
        weighted_f1s = []

        niter = 0
        last_state = None
        while not learning_algo.done():
            cost, state = learning_algo.step()
            last_state = state
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
                print("Iteration {:>5}, accuracy {:4.3f},"
                      "weighted f1 {:4.3f}, topic: '{}', depth: {}"
                      .format(niter, accuracy, weighted_f1, node, depth))

                n_explored.append(len(list(state.state().explored_nodes())))
                n_candidate.append(len(list(state.state().candidate_nodes())))
                n_total.append(len(state.state()._node_info.keys()))
            niter += 1
        return last_state
