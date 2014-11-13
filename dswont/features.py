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

from dswont import search_space as sspace


################################################################################
## Abstraction of IFeature and various useful implementation
################################################################################

class IFeature(metaclass=abc.ABCMeta):
    """ Represents an arbitrary feature function defined on some object types.
    """

    def __init__(self, name: str, description: str=None):
        self._name = name
        self._description = description

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    @abc.abstractmethod
    def compute_one(self, obj):
        return None

    @abc.abstractmethod
    def compute(self, objs):
        return None

    def __call__(self, obj):
        return self.compute_one(obj)


class Feature(IFeature):
    """An implementation of IFeature that is a wrapper around a given function.

    """

    def __init__(self, name: str, feature_fn, description: str=None):
        super().__init__(name, description)
        self._feature_fn = feature_fn

    def feature_fn(self) -> str:
        return self._feature_fn

    def compute_one(self, obj):
        return self._feature_fn(obj)

    def compute(self, objs):
        result = np.fromiter((self._feature_fn(obj) for obj in objs),
                             float,
                             len(objs))
        return result

    def __repr__(self):
        if self._description is None:
            return "Feature('{}')".format(self._name)
        else:
            return "Feature('{}', '{}')".format(self._name, self._description)


class Features(object):
    """Represents a collection of feature functions.

    """

    def __init__(self, *feature_list):
        self._features = collections.OrderedDict()
        for feature_info in feature_list:
            if isinstance(feature_info, IFeature):
                feature_name = feature_info.name()
                feature = feature_info
            else:  # feature_info is a (name, function) tuple
                feature_name = feature_info[0]
                feature = Feature(*feature_info)
            self._features[feature_name] = feature
        self._n_features = len(self._features)

    def feature_names(self):
        return self._features.keys()

    def n_features(self):
        return self._n_features

    def add_feature(self, *args):
        if isinstance(args[0], IFeature):
            assert len(args) == 1
            feature = args[0]
            feature_name = feature.name()
        else:
            feature_name = args[0]
            feature = Feature(*args)
        assert feature_name not in self._features
        self._features[feature_name] = feature
        self._n_features += 1

    def remove_feature(self, name: str):
        assert name in self._features
        self._features.pop(name)
        self._n_features -= 1

    def compute_one(self, obj):
        return np.fromiter((feature.compute_one(obj)
                            for name, feature in self._features.items()),
                           float,
                           self._n_features)

    def compute(self, objs):
        result = np.zeros((len(objs), self._n_features))
        for ftr_idx, feature in enumerate(self._features.values()):
            result[:, ftr_idx] = feature.compute(objs)
        return result

    def value_map(self, obj):
        result = collections.OrderedDict()
        for name, feature in self._features.items():
            result[name] = feature.compute_one(obj)
        return result

    def __call__(self, obj):
        return self.compute_one(obj)

    def __repr__(self):
        feature_repr = (repr(ftr) for ftr in self._features.values())
        return "Features({})".format(', '.join(feature_repr))


class AdditiveSearchStateFeature(IFeature):
    """Represents a feature of the SearchState that is computed incrementally.

    The feature is computed incrementally from the previous SearchState and the
    action. The value of the feature is the value of the previous state plus the
    increment.

    Attributes:
        feature_fn: The function that computes the feature value
            based on the last action, the previous state and its feature value.
            It must have the form feature_fn(action, prev_state, prev_value)
        zero_value: The value of feature on the initial (empty) state.
        delegate: The IFeature object that is used to compute the feature value
            on the previous state.

    """

    def __init__(self, name: str, feature_fn, zero_value=0,
                 description: str=None, delegate=None):
        super().__init__(name, description)
        self._delegate = delegate or self
        self._feature_fn = feature_fn
        self._zero_value = zero_value

    def compute_one(self, sstate: sspace.SearchState):
        prev_state = sstate.previous()
        action = sstate.action()
        if prev_state is None:
            return self._zero_value
        else:
            prev_value = self._delegate.compute_one(prev_state)
            return self._feature_fn(action, prev_state, prev_value)

    def compute(self, sstates):
        compute = self.compute_one
        return [compute(sstate) for sstate in sstates]

    def __call__(self, obj):
        return self.compute_one(obj)

    def __repr__(self):
        if self._description is None:
            return "AdditiveSearchStateFeature('{}')".format(self.name())
        else:
            return "AdditiveSearchStateFeature('{}', '{}')" \
                .format(self.name(), self.description())


class CachingSearchStateFeature(IFeature):
    """Feature of SearchState that caches the results in the state object.

    Attributes:
        delegate: The IFeature object used to compute the feature value.

    """

    def __init__(self, delegate: IFeature):
        self._delegate = delegate

    def name(self):
        return self._delegate.name()

    def description(self):
        return self._delegate.description()

    def _get_create_feature_cache(self, obj):
        feature_cache = getattr(obj, '_feature_cache', None)
        if feature_cache is None:
            feature_cache = {}
            setattr(obj, '_feature_cache', feature_cache)
        return feature_cache

    def _get_value_cached(self, obj, feature_cache):
        feature_name = self._delegate.name()
        value = feature_cache.get(feature_name)
        if value is None:
            value = self._delegate.compute_one(obj)
            feature_cache[feature_name] = value
        else:
            pass
        return value

    def compute_one(self, sstate: sspace.SearchState):
        feature_cache = self._get_create_feature_cache(sstate)
        return self._get_value_cached(sstate, feature_cache)

    def compute(self, objs):
        result = []
        for obj in objs:
            feature_cache = self._get_create_feature_cache(obj)
            result.append(self._get_value_cached(obj, feature_cache))
        return result

    def __call__(self, obj):
        return self.compute_one(obj)

    def __repr__(self):
        return "CachingSearchStateFeature('{}')".format(repr(self._delegate))


class CachingAdditiveSearchStateFeature(CachingSearchStateFeature):
    """The IFeature that relies both on caching and the incremental computation.

    The value is first looked up in the cache, then computed incrementally from
    the previous state and the action. The value for the previous state is first
    looked up in the cache, etc.. The recursion can go down to the initial
    (empty) state, for which the provided zero_value is returned.

    """

    def __init__(self, name: str, incr_feature_fn,
                 zero_value=0, description:str=None):
        delegate = AdditiveSearchStateFeature(name, incr_feature_fn, zero_value,
                                              description, delegate=self)
        super().__init__(delegate)

    def __repr__(self):
        if self.description() is None:
            return "CachingAdditiveSearchStateFeature('{}')".format(self.name())
        else:
            return "CachingAdditiveSearchStateFeature('{}', '{}')" \
                .format(self.name(), self.description())

