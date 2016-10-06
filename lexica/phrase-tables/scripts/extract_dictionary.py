#!/usr/bin/env python3

# This is function to check KL-divergence between
# a conditional distribution and its inverse

from __future__ import division, print_function;
from globalimports import *;
import numpy as np;
import scipy.sparse as sparse;
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

def convex_cleanup(phrase_probs, lex_probs, group_indices):
  size = phrase_probs.size;
  try:
    assert(size == lex_probs.size);
  except AssertionError:
    print("The probability distribution does not have the same size", file=stderr);
    return;
  
  uniq_entries = group_indices.shape[0]
  group_indices = cvx.Constant(group_indices);
  marginals = cvx.Constant(np.ones(uniq_entries));
  sparse_dist = cvx.Variable(size);
  lambd = cvx.Parameter(sign="positive");
  reg   = cvx.norm(sparse_dist, 1);
  phrase_cost = cvx.sum_entries(cvx.kl_div(sparse_dist, phrase_probs));
  lex_cost    = cvx.sum_entries(cvx.kl_div(sparse_dist, lex_probs));
  cost = phrase_cost+lex_cost+reg*lambd;
  constraints = [sparse_dist >= 0, # Positive probability values \
      cvx.sum_entries(group_indices*sparse_dist) == marginals.T, \
      # Sum of probabilities is 1 \
      ];
  for reg_param in np.arange(0.6, 1, 0.01):
    lambd.value = reg_param;
    opt_instance = cvx.Problem(cvx.Minimize(cost), constraints);
    opt_instance.solve();
    yield sparse_dist;

def tester():
  N = 10000;
  pprobs  = np.random.sample(N);
  lprobs  = np.random.sample(N);
  print(type(pprobs));
  print(type(lprobs));
  solutions = convex_cleanup(pprobs, lprobs);
  for sol in solutions:
    new_dist = sol.value;
    print(np.count_nonzero(new_dist), np.min(new_dist), np.max(new_dist));
  return;

def clean_dictionary(phrase_file):
  lexicon = pt.getPhraseEntriesFromTable(phrase_file);
  lexicon = filter(pt.filterLex, lexicon);

  # Make it completely random. Which two distributions we choose to work with
  direction = True if np.random.random() <= 0.5 else False;
  if direction:
    #srctotgt
    entries = list((entry['srcphrase'], \
                    entry['probValues'][0], \
                    entry['probValues'][2]) \
                    for entry in lexicon);
  else:
    #tgttosrc
    entries = list((entry['tgtphrase'], \
                    entry['probValues'][1], \
                    entry['probValues'][3]) \
                    for entry in lexicon);

  # Phrase probs
  pprobs = np.asarray([X[1] for X in entries]);
  lprobs = np.asarray([X[2] for X in entries]);
  # Group by phrase;
  indices = counter();
  vocab = {};
  for X in entries:
    if X[0] not in vocab:
      vocab[X[0]] = next(indices);
  groups = sparse.dok_matrix((len(vocab), len(entries)), dtype=float);
  for idx, entry in enumerate(entries):
    groups[vocab[entry[0]], idx] = 1;
  groups = groups.tocsc();
  print(type(pprobs));
  print(type(lprobs));
  print(type(groups));
  sparse_dists = convex_cleanup(pprobs, lprobs, groups);
  for dist in sparse_dists:
    solution = dist.value;
    print(np.count_nonzero(solution), np.min(solution), np.max(solution));
  return;

if __name__ == '__main__':
  #tester();
  clean_dictionary(sysargv[1]);

