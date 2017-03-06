#!/usr/bin/env python

import numpy as np
from pprint import pprint

class hmm:

    # Q - array of states
    # A - len(Q) x len(Q) transition probability matrix
    # b(w,t) - probability of word w given tag t
    # start_i - index of start state in Q
    def __init__(self, Q, A, b, start_i):
        self.Q = Q
        self.A = A
        self.b = b
        self.start_i = start_i

    # ws - word sequence, array of words
    def decode(self, ws):
        if (len(ws) == 0):
            return []

        viterbi = np.zeros((len(ws), len(self.Q)))
        backpointer = np.zeros((len(ws), len(self.Q)))

        for i in range(0, len(self.Q)):
            viterbi[0][i] = self.A[self.start_i][i] * self.b(ws[0], self.Q[i])
            backpointer[0][i] = self.start_i

        for row in range(1, len(ws)):
            for tcol in range(0, len(self.Q)):
                for fcol in range(0, len(self.Q)):
                    val = viterbi[row-1][fcol] * self.A[fcol][tcol] * self.b(ws[row], self.Q[tcol])
                    if (val > viterbi[row,tcol]):
                        viterbi[row,tcol] = val
                        backpointer[row,tcol] = fcol

        # follow backpointers
        tags = []
        idx = np.argmax(viterbi[len(ws)-1])
        tags.insert(0, self.Q[idx])
        for i in range(len(ws)-1, 0, -1):
            idx = int(backpointer[i,idx])
            tags.insert(0, self.Q[idx])

        return tags
