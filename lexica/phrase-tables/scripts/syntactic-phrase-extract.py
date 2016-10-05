#!/usr/bin/env python3

from __future__ import print_function;

import os, re, sys;
from nltk.tree import Tree;
from globalimports import *;
import random_utils;
import phrasetable_utils;

def generatePhraseSpans(const_tree):
  for subtree in const_tree.subtrees():
    yield (subtree.node, ' '.join(subtree.leaves()));

def extractLexicon(srcsents, tgtsents, alignments, parsetrees=None, weights=None):
  if not weights:
    weights = replicate(1);
  
  # initialize counting tables; 
  vocab_src = defaultdict(lambda: 0.0);
  vocab_tgt = defaultdict(lambda: 0.0);
  ngram_freq_src = defaultdict(lambda: 0.0);
  ngram_freq_tgt = defaultdict(lambda: 0.0);
  lex_table    = defaultdict(lambda: 0.0);
  phrase_table = defaultdict(lambda: 0.0);

  # for each sentence
  for srcsent, tgtsent, alignment, parsetree in zip(srcsents, tgtsents, alignments, parsetrees):
    # preprocessing
    srctokens, tgttokens = re.split('\s+', srcsent), re.split('\s+', tgtsent);
    alignment = re.split('\s+', alignment);
    alignment = [list(map(int, align.split('-'))) for align in alignment];

    # collect counts
    for tok in srctokens:
      vocab_src[tok] += 1;
    for tok in tgttokens:
      vocab_tgt[tok] += 1;
    for align in alignment:
      lex_table[(srctokens[align[0]], tgttokens[align[1]])] += 1;

    # extract phrases 
    aligned_phrases = extractSyntacticPhrases(srctokens, tgttokens, alignment, parsetree, maxPhraseLen=5);
    for phrase in aligned_phrases:
      ngram_freq_src[phrase['srcphrase']] += 1;
      ngram_freq_tgt[phrase['tgtphrase']] += 1;
      phrase_table[(phrase['srcphrase'], phrase['tgtphrase'], phrase['alignments'])] += 1;

  # aggregates over entire table
  for entry in sorted(phrase_table, key=itemgetter(0, 1)):
    yield {'srcphrase': entry[0], \
        'tgtphrase': entry[1], \
        'alignments': entry[2], \
        'probValues': (phrase_table[entry]/ngram_freq_src[entry[0]], \
                       phrase_table[entry]/ngram_freq_tgt[entry[1]], \
                       0, 0), \
        'counts': (phrase_table[entry], ngram_freq_src[entry[0]], ngram_freq_tgt[entry[1]]), \
        }

def extractSyntacticPhrases(srcsent, tgtsent, alignments, bractree=None, \
    maxPhraseLen=7):
  srctokenslen, tgttokenslen = len(srcsent), len(tgtsent);

  alignedTo, alignedFrom = defaultdict(list), defaultdict(list);
  for align in alignments:
    srcidx, tgtidx = align;
    alignedTo[tgtidx].append(srcidx); alignedFrom[srcidx].append(tgtidx);
  unalignedSrcIndices = set(range(srctokenslen))-set(alignedFrom.keys());
  unalignedTgtIndices = set(range(tgttokenslen))-set(alignedTo.keys());
  
  if bractree:
    tree = Tree(bractree.strip());  # In NLTK3, this is Tree.fromstring
    if len(tree.leaves()) != srctokenslen:
      #print("Error: tokenization mismatch between sentence and tree", \
      #    file=stderr);
      #return;
      pass;
    syntacticPhraseIndices = generatePhraseSpans(tree);
  
  # processing phrases in source sentence
  # get possible indices of n-grams in source sentence
  srcPhraseIndices = ((srcidx_start, srcidx_end) \
      for srcidx_start in range(srctokenslen) \
      for srcidx_end in range(srcidx_start, srctokenslen));
  
  # filter them based on length
  srcPhraseIndices = filter(lambda rang: rang[1]+1-rang[0] <= maxPhraseLen,\
      srcPhraseIndices);

  if bractree and False:
    # filter only the syntactic phrases out
    srcPhraseIndices = filter(lambda X: X in syntacticPhraseIndices,\
        srcPhraseIndices);

  for (srcidx_start, srcidx_end) in srcPhraseIndices:
    tgtPhraseIndices = set(tgtidx \
        for idx in range(srcidx_start, srcidx_end+1) \
        for tgtidx in alignedFrom[idx]);
    if not len(tgtPhraseIndices):
      tgtidx_start, tgtidx_end = 0, tgttokenslen-1;
    else:
      tgtidx_start, tgtidx_end = min(tgtPhraseIndices), max(tgtPhraseIndices);

    # Check for out-of-range alignments i.e words should not have alignments
    # outside the windows 
    alignedSrcIndices = set(srcidx \
        for idx in range(tgtidx_start, tgtidx_end+1) \
        for srcidx in alignedTo[idx]);

    if alignedSrcIndices.issubset(set(range(srcidx_start, srcidx_end+1))):
      # no out-of-bounds alignments in source phrase
      # move tgt_min left until you find an aligned word
      # move tgt_max right until you find an aligned word
      for tgtidx_min in range(tgtidx_start, -1, -1):
        for tgtidx_max in range(tgtidx_end, tgttokenslen):
          if tgtidx_max+1-tgtidx_min <= maxPhraseLen:
            alignments = sorted((srcidx, tgtidx) \
                for tgtidx in range(tgtidx_min, tgtidx_max+1) \
                for srcidx in alignedTo[tgtidx]);
            phrase_alignments = tuple('%d-%d' \
                %(srcidx-srcidx_start, tgtidx-tgtidx_start) \
                for srcidx, tgtidx in sorted(alignments));
            yield {'srcphrase': ' '.join(srcsent[srcidx_start:srcidx_end+1]),\
                'tgtphrase': ' '.join(tgtsent[tgtidx_min:tgtidx_max+1]), \
                'alignments': phrase_alignments};
            if tgtidx_max+1 not in unalignedTgtIndices:
              break;
        if tgtidx_min-1 not in unalignedTgtIndices:
          break;
  return;

def main():
  if len(sysargv) < 5:
    print("Error: ./%s <src-file> <tgt-file> <align-file> <src-parses-file>" %(sys.argv[0]), file=stderr);
    sysexit(1);

  srcsents    = random_utils.lines_from_file(sys.argv[1]);
  tgtsents    = random_utils.lines_from_file(sys.argv[2]);
  alignments  = random_utils.lines_from_file(sys.argv[3]);
  parsetrees  = random_utils.lines_from_file(sys.argv[4]);

  for phrase in extractLexicon(srcsents, tgtsents, alignments, parsetrees):
    print(phrasetable_utils.ppphrase(phrase));
  return;

if __name__ == '__main__':
  main();
