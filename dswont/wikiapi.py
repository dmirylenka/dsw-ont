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
import logging
import re
import requests
import semidbm

from dswont import dbpedia

API_URL = 'http://en.wikipedia.org/w/api.php'

HEADERS = {
    'User-Agent': 'Daniil Mirylenka (mirylenka@fbk.eu)'
}

WIKI_SUBCAT_INDEX_FILE = '/Users/dmirylenka/data/dswont/uri-to-subcats'
WIKI_SUPERCAT_INDEX_FILE = '/Users/dmirylenka/data/dswont/uri-to-supercats'


def page(title=None, pageid=None, text=True):
    params = {
        'action': 'query',
        'format': 'json',
        'redirects': ''
    }

    if title:
        params['titles'] = title
    elif pageid:
        params['pageids'] = pageid
    else:
        raise ValueError('Both page title and pageid are empty.')

    if text:
        params['prop'] = 'extracts'
        params['explaintext'] = ''

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    page = list(data['query']['pages'].values())[0]

    if page.get('missing') == '':
        logging.warning('Wikipedia page not found: title={}, pageid={}'
                        .format(title, pageid))
        return None

    return {
        'title': page['title'],
        'pageid': page['pageid'],
        'text': page.get('extract')
    }


def supercats(uri):
    title = dbpedia.uri_to_title(uri)

    params = {
        'action': 'query',
        'prop': 'categories',
        'format': 'json',
        'titles': 'Category:{}'.format(title),
        'clshow': '!hidden'
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    page = list(data['query']['pages'].values())[0]

    if page.get('missing') == '':
        logging.warning('Wikipedia page not found: title={}'
                        .format(title))
        return None

    def get_title(cat):
        return dbpedia.title_to_uri(re.sub('^Category:', '', cat['title']),
                                    category=True)

    if 'categories' in page:
        return list(map(get_title, page['categories']))
    else:
        logging.warning('Category: {} has no parent categories.'
                        .format(title))
        return []


def subcats(uri):
    title = dbpedia.uri_to_title(uri)

    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': 'Category:{}'.format(title),
        'cmtype': 'subcat',
        'cmlimit': '500',
        'format': 'json'
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    categories = list(data['query']['categorymembers'])

    def get_uri(cat):
        return dbpedia.title_to_uri(re.sub('^Category:', '', cat['title']),
                                    category=True)

    return list(map(get_uri, categories))


class WikipediaGraphIndex(object):
    """Provides navigation over the Wikipedia category graph.

    Returns the subcategories and the super-categories of a given category.
    The data is retrieved from Wikipedia and cached into a database.
    The categories are represented by their DBpedia URIs, such as, for example,
    http://dbpedia.org/resource/Category:Mac_OS_X_backup_software .

    You should open() WikipediaGraphIndex object before using and close() after,
    or otherwise use it with a 'with' clause, as below:

    Use:
        category = 'http://dbpedia.org/resource/Category:Computer_science'
        with WikipediaGraphIndex() as wiki:
            print(wiki.get_subcats(category))

    """

    def __init__(self,
                 subcat_index_file=WIKI_SUBCAT_INDEX_FILE,
                 supercat_index_file=WIKI_SUPERCAT_INDEX_FILE):
        self._subcat_index_file = subcat_index_file
        self._supercat_index_file = supercat_index_file

    def _get_related_topics(self, topic, relation, cache, api_get):
        """Gets the topics related to the given topic, e.g. subcats or
        supercats."""
        if topic.encode('utf-8') in cache:
            related = cache[topic].decode('utf-8').split()
        else:
            related = api_get(topic)
            if related is None:
                logging.warning('Page not in Wikipedia, perhaps deleted: {}'
                                .format(dbpedia.uri_to_title(topic)))
                related = []
            cache[topic] = ' '.join(related)
        return related

    def get_subcats(self, topic):
        return self._get_related_topics(topic, 'subcat',
                                        self._subcat_index, subcats)

    def get_supercats(self, topic):
        return self._get_related_topics(topic, 'supercat',
                                        self._supercat_index, supercats)

    def open(self):
        self._subcat_index = semidbm.open(self._subcat_index_file, 'c')
        self._supercat_index = semidbm.open(self._supercat_index_file, 'c')
        return self

    def close(self):
        self._subcat_index.close()
        self._supercat_index.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False


DEFAULT_EXCLUSION_PATTERNS = [
        lambda title: title.endswith(" portal"),
        lambda title: title.endswith(" stubs"),
        lambda title: title.startswith("Wikipedia ")
]


class CategoryRelationCache(object):
    """ The in-memory cache for the parent-child category relations.

    You should open() CategoryRelationCache before using, and close()
    after, or use it with a 'with' clause.

    """

    def __init__(self,
                 subcat_index_file=WIKI_SUBCAT_INDEX_FILE,
                 supercat_index_file=WIKI_SUPERCAT_INDEX_FILE,
                 exclusion_fns=DEFAULT_EXCLUSION_PATTERNS):
        self._wiki_graph = WikipediaGraphIndex(
            subcat_index_file=subcat_index_file,
            supercat_index_file=supercat_index_file)
        self._children = collections.OrderedDict()
        self._parents = collections.OrderedDict()
        exclusion_fns = exclusion_fns or []
        def filter_fn(title):
            return not any(exclude(title) for exclude in exclusion_fns)
        self.filter_fn = filter_fn

    def _compute_parents(self, node):
        supercat_uris = self._wiki_graph. \
            get_supercats(dbpedia.to_category_uri(node))
        supercat_titles = (dbpedia.to_title(uri) for uri in supercat_uris)
        return sorted(filter(self.filter_fn, supercat_titles))

    def _compute_children(self, node):
        subcat_uris = self._wiki_graph. \
            get_subcats(dbpedia.to_category_uri(node))
        subcat_titles = (dbpedia.to_title(uri) for uri in subcat_uris)
        return sorted(filter(self.filter_fn, subcat_titles))

    def parents(self, node):
        if node in self._parents:
            return self._parents[node]
        else:
            result = self._compute_parents(node)
            self._parents[node] = result
            return result

    def children(self, node):
        if node in self._children:
            return self._children[node]
        else:
            result = self._compute_children(node)
            self._children[node] = result
            return result

    def open(self):
        self._wiki_graph.open()
        return self

    def close(self):
        self._wiki_graph.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

