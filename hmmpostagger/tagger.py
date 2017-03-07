#!/usr/bin/env python

import numpy as np
import argparse
from pprint import pprint
from hmm import hmm

class tagset:
    def __init__(self):
        self.taglist = []
        self.tagindex = {}

    def add(self, tag):
        if tag not in self.tagindex:
            self.taglist.append(tag)
            self.tagindex[tag] = len(self.taglist) - 1

    def getindex(self, tag):
        return self.tagindex[tag]

    def list(self):
        return self.taglist

class countmap2:
    def __init__(self):
        self.m = {}

    def incr(self, a, b):
        if a not in self.m:
            self.m[a] = {}
        if b not in self.m[a]:
            self.m[a][b] = 0
        self.m[a][b] += 1

    def normalize(self):
        for a in self.m:
            mass = sum(self.m[a].values())
            for b in self.m[a]:
                self.m[a][b] /= float(mass)

    def get(self, a, b):
        return self.m[a][b]

class tagger:
    def train(self, files):
        self.tagset = tagset()
        self.tagset.add('start')
        self.transition_freq = countmap2()
        self.word_freq = countmap2()

        for f in files:
            fhandle = open(f, 'r')
            for line in fhandle:
                if line != '\n':
                    prev_pos = 'start'
                    for token in line.split(' '):
                        parts = token.strip().split('/')
                        if len(parts) == 2:
                            word = parts[0]
                            pos = parts[1]

                            self.tagset.add(pos)
                            self.transition_freq.incr(prev_pos, pos)
                            self.word_freq.incr(word, pos)
                            prev_pos = pos

            fhandle.close()

        self.transition_freq.normalize()
        self.word_freq.normalize()

        self.Q = self.tagset.list()
        self.A = np.zeros((len(self.Q), len(self.Q)))
        for i in range(0, len(self.Q)):
            for j in range(0, len(self.Q)):
                try:
                    self.A[i,j] = self.transition_freq.get(self.Q[i], self.Q[j])
                except KeyError:
                    pass

        def _b(word, pos):
            try:
                return self.word_freq.get(word, pos)
            except KeyError:
                return 0
        self.b = _b

        self.hmm = hmm(self.Q, self.A, _b, 0)

    def tag(self, words):
        return self.hmm.decode(words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMM for POS tagging')
    parser.add_argument('--trainingdata', type=str, nargs='+', required=True)
    parser.add_argument('--words', type=str, nargs='+', required=True)
    args = parser.parse_args()

    tg = tagger()
    tg.train(args.trainingdata)
    tags = tg.tag(args.words)

    print ' '.join(map(lambda wp : wp[0] + "/" + wp[1], zip(args.words, tags)))
