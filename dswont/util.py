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

import json
import logging
import nltk
import numpy as np
import os
import pandas as pd
import re
import shutil
from sklearn import metrics


## Magic for reading a single-character input from the command line,
## that is reading user keystrokes. Found somewhere on the internet.
def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt

        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty

    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch


getch = _find_getch()


def resource(filename):
    """Returns the absolute path to the resource file.

    """
    project_dir = os.path.dirname(os.path.dirname(__file__))
    module_dir = os.path.join(project_dir, 'dswont')
    resource_dir = os.path.join(module_dir, "resources")
    return os.path.join(resource_dir, filename)


class PersistentDict(dict):
    def __init__(self, filename, *args, **kwds):
        self.filename = filename
        if os.access(filename, os.R_OK):
            fileobj = open(filename, 'r')
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwds)

    def sync(self):
        """Write dict to disk.

        """
        filename = self.filename
        tempname = filename + '.tmp'
        fileobj = open(tempname, 'w')
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)  # atomic commit

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        json.dump(self, fileobj, separators=(',', ':'))

    def load(self, fileobj):
        fileobj.seek(0)
        try:
            return self.update(json.load(fileobj))
        except Exception as e:
            logging.warning('Exception while loading the file: ' + e)


stopwords = {line.rstrip()
             for line in open(resource('en-stopwords.txt')).readlines()}

words_re = re.compile('\w+')


def pos_tag(string_or_tokens: str):
    if isinstance(string_or_tokens, str):
        string = string_or_tokens
        tokens = nltk.tokenize.word_tokenize(string)
    else:
        tokens = string_or_tokens
    return nltk.pos_tag(tokens)


def head_word_pos(phrase_or_pos_tagging):
    if isinstance(phrase_or_pos_tagging, str):
        phrase = phrase_or_pos_tagging
        #         phrase_wo_parens = re.search(r'^(.*?)( \(.*\))?$', phrase)
        # .groups()[0]
        phrase_wo_parens = re.sub('\s\(.*\)', '', phrase)
        pos_tagging = pos_tag(phrase_wo_parens)
    else:
        pos_tagging = phrase_or_pos_tagging

    pos_to_skip = {"VBN", "VBD"  #, "CD"
    }
    delimiting_pos = {"DT", "IN"}
    delimiting_words = {"("}
    result = None
    for word, pos in pos_tagging:
        current_word_is_delimiter = pos in delimiting_pos or word in \
                                    delimiting_words
        nondelimiters_encountered = result is not None
        if current_word_is_delimiter and nondelimiters_encountered:
            break
        elif pos not in pos_to_skip:
            result = word, pos
    if not result:
        raise Exception(
            "Could not find the head word of the phrase '{}'".format(
                pos_tagging))
    return result


def measure_params_ytrue_ypred_(*params):
    if len(params) == 2:
        y_true, y_pred = params
    elif len(params) == 3:
        estimator, X, y_true = params
        y_pred = estimator.predict(X)
    else:
        raise ValueError(
            "weighted_f1 called with {} parameters, \
            the correct signature is:\n weighted_f1(y_true, y_pred) \
            or weighted_f1(estimator, X, y_true).".format(len(params)))
    return np.array(y_true, dtype=bool), np.array(y_pred, dtype=bool)


def weighted_f1(*params):
    y_true, y_pred = measure_params_ytrue_ypred_(*params)
    score1 = metrics.f1_score(y_true, y_pred, pos_label=1)
    score2 = metrics.f1_score(y_true, y_pred, pos_label=0)
    nones = sum(y_true)
    nzeros = len(y_true) - nones
    return (nones * score1 + nzeros * score2) / (nones + nzeros)


def f1_pos_class(*params):
    y_true, y_pred = measure_params_ytrue_ypred_(*params)
    return metrics.f1_score(y_true, y_pred, pos_label=1)


def f1_neg_class(*params):
    y_true, y_pred = measure_params_ytrue_ypred_(*params)
    return metrics.f1_score(y_true, y_pred, pos_label=0)


def accuracy_score(*params):
    y_true, y_pred = measure_params_ytrue_ypred_(*params)
    return metrics.accuracy_score(y_true, y_pred)


def orgtable(df: pd.DataFrame):
    return [list(row.values) for idx, row in df.iterrows()]