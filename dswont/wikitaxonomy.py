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

import pandas as pd
import warnings

from dswont import dbpedia

warnings.filterwarnings("ignore", category=DeprecationWarning)

WTX_NODE_TYPE_FILE = '/Users/dmirylenka/data/wikitaxonomy/node-types.txt'
WTX_NODE_REL_FILE = '/Users/dmirylenka/data/wikitaxonomy/rel-types.txt'


def nodes_data(file):
    data = pd.read_csv(file, sep=' ', names=['node', 'type'])
    data['node'] = data['node'].str.replace('_', ' ')
    data = data.set_index('node')
    data['is_class'] = data['type'].apply(lambda x: 'class' == x)
    return data


def rel_data(file):
    data = pd.read_csv(file, sep=' -> |\s', names=['parent', 'child'])
    data['parent'] = data['parent'].str.replace('_', ' ')
    data['child'] = data['child'].str.replace('_', ' ')
    data = data.set_index(['parent', 'child'])
    return data


def to_title(title_or_uri):
    if dbpedia.is_category_uri(title_or_uri):
        return dbpedia.uri_to_title(title_or_uri)
    else:
        return title_or_uri


class WikiTaxonomy(object):
    def __init__(self, nodes_file=WTX_NODE_TYPE_FILE,
                 rel_file=WTX_NODE_REL_FILE):
        self._nodes = nodes_data(nodes_file)
        self._rels = rel_data(rel_file)

    def contains(self, node):
        return to_title(node) in self._nodes.index

    def is_class(self, node):
        node = to_title(node)
        assert self.contains(node)
        return self._nodes.loc[node, 'is_class']

    def has_rel(self, parent, child):
        parent = to_title(parent)
        child = to_title(child)
        assert self.contains(parent)
        assert self.contains(child)
        return (parent, child) in self._rels.index

