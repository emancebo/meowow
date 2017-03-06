#!/usr/bin/env python

import sys
sys.path.append("..")

from hmmpostagger.hmm import hmm
from pprint import pprint

w = ["big", "planes", "fly", "fast"]
Q = ["start", "noun", "verb", "adj", "adv"]
A = [[0, 0.7, 0, 0.3, 0],
    [0, 0, 0.6, 0.1, 0.3],
    [0, 0, 0.3, 0.3, 0.4],
    [0, 0.8, 0, 0.2, 0],
    [0, 0.2, 0.7, 0.1, 0]]

B = {
    "big": {
        "adj": 0.8,
        "adv": 0.2
    },
    "planes": {
        "noun": 0.7,
        "verb": 0.3
    },
    "fly": {
        "noun": 0.3,
        "verb": 0.5,
        "adj": 0.2
    },
    "fast": {
        "adv": 1.0
    }
}

def b(word, tag):
    try:
        return B[word][tag]
    except KeyError:
        return 0

h = hmm(Q, A, b, 0)
pprint(w)
pprint(h.decode(w))
