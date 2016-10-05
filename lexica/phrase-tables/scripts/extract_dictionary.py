#!/usr/bin/env python3

# This is function to check KL-divergence between
# a conditional distribution and its inverse

from __future__ import division, print_function;
import sys;

import numpy as np;
import cvxpy as cvx;

import phrasetable_utils as pt;


def test():
  from collections import Counter;
  from itertools import product;
  from math import log;
  from operator import itemgetter;
  import string;

  def simulate_KL(space=1000, tries=20):
    alnum = string.ascii_letters + string.digits;
    printable = string.printable;
    population = list(product(alnum, alnum, printable));
    kldivergence = lambda X, Y: sum(\
        x*log(x)-x*log(y) for x,y in zip(X, Y) );

    for trial in range(tries):
      random.shuffle(population);
      samples    = random.sample(population, space);
      trigrams   = Counter(samples);
      forwardmarginals   = Counter(map(itemgetter(0,1), samples));
      backwardmarginals  = Counter(map(itemgetter(2), samples));
      forwardconddist    = [trigrams[word]/forwardmarginals[word[:-1]] \
          for word in sorted(trigrams.keys())];
      backwardconddist   = [trigrams[word]/backwardmarginals[word[-1]] \
          for word in sorted(trigrams.keys())];

      dist = kldivergence(forwardconddist, backwardconddist);
      print(dist, dist/space);

  for space in (10000, 100000, 10000):
    simulate_KL(space=space);
  return;

def convex_cleanup(phrase_probs, lex_probs):
  size = phrase_probs.size;
  try:
    assert(size == lex_probs.size);
  except AssertionError:
    print("The probability distribution does not have the same size", file=stderr);
    return;

  sparse_dist = cvx.Variable(size);
  lambd = cvx.Parameter(sign="positive");
  reg   = cvx.norm(sparse_dist, 1);

  phrase_cost = cvx.sum_entries(cvx.kl_div(sparse_dist, phrase_probs));
  lex_cost    = cvx.sum_entries(cvx.kl_div(sparse_dist, lex_probs));
  cost = phrase_cost+lex_cost+lambd*reg;

  for reg_param in np.arange(0.6, 1, 0.01):
    lambd.value = reg_param;
    opt_instance = cvx.Problem(cvx.Minimize(cost), \
        [sparse_dist >= 0, # Positive probability values \
        ]);
    opt_instance.solve();
    yield sparse_dist;

def tester():
  N = 10000;
  pprobs  = np.random.sample(N);
  lprobs  = np.random.sample(N);
  solutions = convex_cleanup(pprobs, lprobs);
  for sol in solutions:
    new_dist = sol.value;
    print(np.count_nonzero(new_dist), np.min(new_dist), np.max(new_dist));
  return;

def clean_dictionary(phrase_file):
  lexicon = pt.getPhraseEntriesFromTable(phrase_file);
  lexicon = pt.getLexiconEntries(phrase_file);

  # Make it completely random. Which two distributions we choose to work with
  direction = srctotext == True if np.random.random() <= 0.5 else False;
  if direction:
    entries = list((entry['srcphrase'], \
                    entry['probValues'][0], \
                    entry['probValues'][2]) \
                  for entry in lexicon);
  else:
    entries = list((entry['tgtphrase'], \
                    entry['probValues'][1], \
                    entry['probValues'][3]) \
                  for entry in lexicon);
  return;

if __name__ == '__main__':
  #tester();
  clean_dictionary(sys.argv[1]);

