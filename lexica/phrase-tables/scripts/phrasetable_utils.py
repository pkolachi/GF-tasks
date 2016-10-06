#!/usr/bin/env python

from __future__ import print_function;
import math, operator, string; 

from globalimports import *;
import random_utils;

def parseentry(entry, \
    pfields=['srcphrase', 'tgtphrase', 'probValues', 'alignments', 'counts'],\
    delimiter='|||'):
  fields = entry.split(delimiter);
  phrase = defaultdict(lambda: '');
  for field, value in zip(pfields, \
      map(string.strip, filter(None, fields))):
    if field == 'probValues':
      phrase[field.strip()] = tuple(map(float, value.split()));
    elif field == 'alignments':
      phrase[field.strip()] = tuple(value.split());
    elif field == 'counts':
      phrase[field.strip()] = tuple(map(float, value.split()));
    elif field in ['logprob', 'pmival', 'pmivalvar']:
      phrase[field.strip()] = float(value.strip());
    else:
      phrase[field.strip()] = value.strip();
  return phrase;

def ppphrase(entry, fields=None, delim='\t'):
  output_values = [];
  outformat = lambda f, v: ' '.join(str(x) for x in v) \
                 if f == 'probValues' \
                 else ' '.join(str(int(x)) for x in v) if f == 'counts' \
                 else ' '.join(v) if f == 'alignments' \
                 else repr(v) if v else "NONE";
  if fields == None:
    output_values = [ entry['srcphrase'], entry['tgtphrase'], \
        outformat('probValues', entry['probValues']), \
        outformat('alignments', entry['alignments']), \
        outformat('counts', entry['counts']) ];
  else:
    output_values = [entry['srcphrase'], entry['tgtphrase']] + \
                    [outformat(f, entry[f]) for f in fields \
                        if f not in ['srcphrase', 'tgtphrase']]; 
  return delim.join(output_values);

def getPhraseEntriesFromTable(phraseTableFile, entryParser=None):
  entryParser = parseentry if not entryParser else entryParser;
  return map(entryParser, random_utils.lines_from_file(phraseTableFile));
  #return (entryParser(line) \
  #    for line in random_utils.lines_from_file(phraseTableFile));

def getPhraseEntriesFromTable_alt(phraseTableFile, entryParser=None):
  entryParser = parseentry if not entryParser else entryParser;
  for line in random_utils.lines_from_file(phraseTableFile):
    yield entryParser(line.strip());

def par_getPhraseEntriesFromTable(phraseTableFile, entryParser=None, cores=0):
  entryParser = parseentry if not entryParser else entryParser;
  cores = multiprocessing.cpu_count() if not cores else cores;
  with multiprocessing.Pool(cores) as jobs:
    return jobs.imap_unordered(entryParser, \
        random_utils.lines_from_file(phraseTableFile), chunksize=100000);

def getPhraseBlocksFromTable(phraseTableFile, entryParser=None):
  count = 0;
  entryParser = parseentry if not entryParser else entryParser;
  '''
  phraseBlocks = {};
  srcPhrases = {};
  for phraseEntry in imap(entryParser, random_utils.lines_from_file(phraseTableFile)):
	if phraseBlocks.has_key( phraseEntry['srcphrase'] ):
      phraseBlocks[phraseEntry['srcphrase']].append(phraseEntry);
	else:
      for srcphrase, entries in phraseBlocks.iteritems():
        yield entries;
      phraseBlocks = {};
      phraseBlocks.setdefault(phraseEntry['srcphrase'], []).append(phraseEntry);
  '''

def getLexiconEntries(phraseTableFile, entryParser=None):
  entryParser = parseentry if not entryParser else entryParser;
  fields = ['srcphrase', 'tgtphrase', 'logprob', 'pmival', 'pmivalvar'];
  return map(entryParser, random_utils.lines_from_file(phraseTableFile), replicate(fields), replicate('\t'));

def pruneTableByLogProb(tableIter, paramValues, K=10000):
  probminValue, probmaxValue = 0, 1;
  bucketCountForSort = 20;
  bucketWidth = float(probmaxValue-probminValue)/bucketCountForSort;
  phraseBuckets = defaultdict(list);
  phraseCount = 0;
  for entry in tableIter:
    entry['logprob'] = sum(map(operator.mul, map(math.log, entry['probValues']), paramValues));
	#phrasePair = ( entry['srcphrase'], entry['tgtphrase'], sum(map(operator.mul, map(math.log10, entry['probValues']), paramValues)) );
    yield entry;
  '''
    bucketIdx = int(phrasePair[2]/bucketWidth)+1;
    phraseBuckets[bucketIdx].append(phrasePair);
  for bucketIdx in sorted(phraseBuckets.keys(), reverse=True):
	for entry in sorted(phraseBuckets[bucketIdx], key=lambda x: x[2], reverse=True):
      yield entry;
  '''
def moses_detokenize(token):
  token = token.replace('&amp;', '&');
  token = token.replace('_PIPE_', '|');
  token = token.replace('&lt;', '<');
  token = token.replace('&gt;', '>');
  token = token.replace('&apos;', "'");
  token = token.replace('&quot;', '"');
  token = token.replace('-lrb', '(');
  token = token.replace('-rrb', ')');
  return token;

# Filters for choosing entries in phrase-table
# no pruning at all
filterP   = lambda phraseEntry: True ; 
# pruning for lexicon (1-grams) 
filterLex = lambda phraseEntry: \
    (len(phraseEntry['srcphrase'].split()) == 1 or \
    len(phraseEntry['tgtphrase'].split()) == 1);
# pruning for multi-word expressions (3-grams) 
filterMWE = lambda phraseEntry: \
    (len(phraseEntry['srcphrase'].split()) >= 3 or \
    len(phraseEntry['tgtphrase'].split()) >= 3);

