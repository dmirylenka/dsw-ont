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
from dswont import dbpedia
import re
import util

def is_acm_id(string: str):
    return re.match('^\d{8}$', string) is not None


class AcmWrapper(object):
    def _read_concepts(self, filename):
        concepts = {}
        with open(filename) as f:
            for line in f.readlines():
                id, name = line.strip().split('\t')
                concepts[id] = name
        return concepts

    def _read_mapping(self, filename):
        pairs = []
        with open(filename) as f:
            for line in f.readlines():
                acm_id, wiki_title = line.strip().split('\t')
                pairs.append((acm_id, dbpedia.to_category_uri(wiki_title)))
        return zip(*pairs)

    def _read_relations(self, filename):
        children = collections.defaultdict(set)
        with open(filename) as f:
            for line in f.readlines():
                parent_id, child_id = line.strip().split('\t')
                children[parent_id].add(child_id)
        return children

    def __init__(self,
                 acm_concept_file=util.resource("acm-concepts.txt"),
                 acm_rels_file=util.resource("acm-relations.txt"),
                 acm_mapping_file=util.resource("acm-wiki-mapping.txt")):

        self._concepts = self._read_concepts(acm_concept_file)

        acm_ids, wiki_uris = self._read_mapping(acm_mapping_file)
        self._wiki2acm = dict(zip(wiki_uris, acm_ids))
        self._acm2wiki = {v: k for k, v in self._wiki2acm.items()}

        self._children = self._read_relations(acm_rels_file)

    def contains(self, node):
        if is_acm_id(node):
            return node in self._acm2wiki
        else:
            return dbpedia.to_category_uri(node) in self._wiki2acm

    def _to_wiki(self, node):
        return self._acm2wiki[node]

    def _to_acm(self, node):
        return self._wiki2acm[dbpedia.to_category_uri(node)]

    def children(self, node):
        if self.contains(node):
            acm_id = self._to_acm(node)
            children = self._children[acm_id]
            return [self._to_wiki(child)
                    for child in children
                    if self.contains(child)]
        else:
            return set()

    def is_parent(self, parent, child):
        return dbpedia.to_category_uri(child) in self.children(parent)

    def is_ancestor(self, parent, child):
        if self.is_parent(parent, child):
            return True
        return any([self.is_ancestor(parents_child, child)
                    for parents_child in self.children(parent)])
