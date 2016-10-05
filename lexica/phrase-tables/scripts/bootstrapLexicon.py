#!/usr/bin/env python3

from globalimports import *;
import random_utils, conll_utils;

import codecs, collections, cProfile, itertools, pstats, sys; 
from phrasetable_utils import *;

def collectSrcPhraseCounts(phraseTableIter):
  counts, totalCount = defaultdict(lambda: 0.0), 0;
  for phraseEntry in phraseTableIter:
    counts[phraseEntry['srcphrase']] += phraseEntry['counts'][0];
    totalCount += phraseEntry['counts'][0];
  return counts, totalCount; 

def getPhrasePairStrengths(phrasetable, paramValues):
  # COMPUTE POINT-WISE MUTUAL INFORMATION
  # PICK POSSIBLE TRANSLATIONS BASED ON PMI VALUES
  srcPhraseCounts, totalSrcPhraseCounts = \
      collectSrcPhraseCounts(getPhraseEntriesFromTable(phrasetable));
  for srcphrase in srcPhraseCounts:
    srcPhraseCounts[srcphrase] = srcPhraseCounts[srcphrase]/totalSrcPhraseCounts;
  for phraseEntry in pruneTableByLogProb(getPhraseEntriesFromTable(phrasetable), paramValues):
    srcphrase = phraseEntry['srcphrase'];
    srcphraseProb = srcPhraseCounts[srcphrase]/totalSrcPhraseCounts;
    phraseEntry['pmival'] = math.log(math.exp(phraseEntry['logprob'])/srcPhraseCounts[srcphrase], 2);
    phraseEntry['pmivalvar'] = phraseEntry['counts'][2]*phraseEntry['pmival'];
    yield phraseEntry;

def getPhraseTable(args):
  # parse param values; 
  phrasetablefile = args[0];
  paramValues = map(float, args[1].strip().split());
  for phrasePair in pruneTableByLogProb(ifilter(filterP, \
      getPhraseEntriesFromTable(phrasetablefile)), paramValues):
    print ppphrase(phrasePair, form='long', fields=['pmivalvar']);
  return;

def getLexicon(args):
  # parse param values; 
  phrasetablefile = args[0];
  paramValues = map(float, args[1].strip().split());
  for phrasePair in ifilter(filterLex, \
      getPhrasePairStrengths(phrasetablefile, paramValues)):
    print ppphrase(phrasePair, fields=['probValues', 'pmival', 'pmivalvar']);
  return;

def getCleanPhraseEntries(lexicontablefile):
  alphaChar = lambda x: x not in string.punctuation and x not in string.digits;
  puncChar  = lambda x: x not in '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
  lenFilter = lambda token: len(token.split()) > 0;
  #categoryFilter = lambda tagsList: set.intersection(set(tagsList), set(['NN', 'NNS', 'NNP', 'NNPS']));
  #categoryFilter = lambda tagsList: not set.intersection(set(tagsList), set(['CC', 'CD', 'DT', 'FW', 'IN', 'MD', 'RP', 'SYM', 'TO', 'UH']));
  categoryFilter  = lambda x: True;
  
  for line in random_utils.lines_from_file(lexicontablefile):
    srcphrase, tgtphrase, values = line.strip().split('\t', 2);
    srcphrase = moses_detokenize(srcphrase);
    if categoryFilter(srcphrase) and filter(alphaChar, srcphrase).strip():
      srcphraseRepr = ' '.join( filter(puncChar, srcphrase).split() ).strip(' -');
      if not lenFilter(srcphraseRepr):
        continue;
    else:
      continue;
    tgtphrase = moses_detokenize(tgtphrase);
    if categoryFilter(tgtphrase) and filter(alphaChar, tgtphrase).strip():
      tgtphraseRepr = ' '.join( filter(puncChar, tgtphrase).split() ).strip(' -');
      if not lenFilter(tgtphraseRepr):
        continue;
    else:
      continue;
    yield '%s\t%s\t%s' %(srcphraseRepr, tgtphraseRepr, values);

def getStatistics(args):
  from correlations import rank_spearmansp, kendalls_tau;
  import numpy as np;
  import numpy.random as rand;
  import matplotlib.pyplot as plot;
  
  lexiconfile, plotfile = args[0], args[1];
  logprobList, pmiList, localpmiList = [], [], [];
  entryType, entry_count = {}, 0;
  transMem = {};
  for phraseEntry in getCleanPhraseEntries(lexiconfile):
    srcphrase, tgtphrase, values = phraseEntry.strip().split('\t', 2);
    entryClass = 0 if len(srcphrase.split()) == len(tgtphrase.split()) else 1 if len(srcphrase.split()) > len(tgtphrase.split()) else 2;
    if transMem.has_key( (srcphrase, tgtphrase) ):
      continue;
    else:
      transMem[ (srcphrase, tgtphrase) ] = True;
    if entryClass == 2:
      print srcphrase, tgtphrase, len(srcphrase.split()), len(tgtphrase.split()), entry_count
    entryType.setdefault(entryClass, []).append(entry_count);
    logprob, pmi, localpmi = map(float, values.split('\t'));
    logprobList.append( logprob );
    pmiList.append( pmi );
    localpmiList.append( localpmi );
    entry_count += 1;

  print >>sys.stderr, "Computing correlations between entry scores";
  for i in itertools.combinations([1,2,3], 2):
    X_name = 'logprobList' if i[0] == 1 else ('pmiList' if i[0] == 2 else 'localpmiList');
    Y_name = 'logprobList' if i[1] == 1 else ('pmiList' if i[1] == 2 else 'localpmiList');
    X = logprobList if i[0] == 1 else (pmiList if i[0] == 2 else localpmiList);
    Y = logprobList if i[1] == 1 else (pmiList if i[1] == 2 else localpmiList);
    print X_name, Y_name, rank_spearmansp(X, Y), kendalls_tau(X, Y);
    print >>sys.stderr, "Length wise statistics";
    fig = plot.figure();
    linestyles = {0: '_', 1: '--', 2: '-'};
    sampleSize = sys.maxint;
  
  for entryClass in entryType:
    classmembers  = np.asarray(entryType[entryClass][:sampleSize], dtype='int');
    classlogprob  = np.asarray([logprobList[idx] for idx in classmembers], dtype='float64');
    classpmi      = np.asarray([pmiList[idx] for idx in classmembers], dtype='float64');
    classlocalpmi = np.asarray([localpmiList[idx] for idx in classmembers], dtype='float64');
    center_classlogprob = (classlogprob-np.min(classlogprob))/(np.max(classlogprob)-np.min(classlogprob));
    center_classpmi = (classpmi-np.min(classpmi))/(np.max(classpmi)-np.min(classpmi));
    center_classlocalpmi = (classlocalpmi-np.min(classlocalpmi))/(np.max(classlocalpmi)-np.min(classlocalpmi));
    threshold_value = np.percentile(center_classlocalpmi, 60);
    print entryClass, len(classmembers), np.mean(center_classlocalpmi), np.var(center_classlocalpmi), threshold_value;
    
    #plot.plot(classmembers, center_classlogprob, color='b', linestyle=linestyles[entryClass], label='Log probability for class %d'%(entryClass));
	#plot.plot(classmembers, center_classpmi, color='r', linestyle=linestyles[entryClass], label='Pmi for class %d'%(entryClass));
	#plot.plot(classmembers, center_classlocalpmi, color='k', linestyle=linestyles[entryClass], label='Local pmi for class %d'%(entryClass));
    #plot.savefig(plotfile+'.png', format='png');
  return;

def pruneLemmasList(lemmas):
  return sorted(lemmas, key=lambda args:args[-1])[0][0];

def getGFLexicon(args):
  import numpy as np;
  import pgf;
  
  #lexiconfile, srcpgffile, tgtpgffile = args[0], args[1], args[2];
  lexiconfile, pgffile = args[0], args[1];
  
  logprobList, pmiList, localpmiList = [], [], [];
  entryType, entry_count = {}, 0;
  transMem = {};
  minpmi, maxpmi = sys.maxint, (-2**30);
  for phraseEntry in getCleanPhraseEntries(lexiconfile):
    srcphrase, tgtphrase, values = phraseEntry.strip().split('\t', 2);
    entryClass = 0 if len(srcphrase.split()) == len(tgtphrase.split()) else 1 if len(srcphrase.split()) > len(tgtphrase.split()) else 2;
    if transMem.has_key( (srcphrase, tgtphrase) ):
      continue;
    entryType.setdefault(entryClass, []).append(entry_count);
    logprob, pmi, localpmi = map(float, values.split('\t'));
    logprobList.append( logprob );
    pmiList.append( pmi );
    localpmiList.append( localpmi );
    transMem[ (srcphrase, tgtphrase) ] = localpmi;
    minpmi = localpmi if localpmi < minpmi else minpmi;
    maxpmi = localpmi if localpmi > maxpmi else maxpmi;
    entry_count += 1;

  localpmiThreshold = np.percentile( np.asarray(localpmiList, dtype='float64'), 40 );
  minpmi = localpmiThreshold;
  #print localpmiThreshold;
  
  grammar = pgf.readPGF(pgffile);
  srcmorphanalyzer, tgtmorphanalyzer = grammar.languages['TranslateEng'].lookupMorpho, grammar.languages['TranslateSwe'].lookupMorpho;
  morphRepr = lambda morphObj: morphObj[0];
  count = 0;
  for phraseEntry in filter(lambda entry: True if entry[1] > localpmiThreshold else False, sorted(transMem.iteritems(), key=lambda (k,v):(v,k), reverse=True)):
    count += 1;
    pmiValue = phraseEntry[1];
    normpmiValue = (pmiValue-minpmi)/(maxpmi-minpmi);
    #print phraseEntry[0], type(phraseEntry[0][0]), type(phraseEntry[0][1]);
    try:
      srcLemmas = [anal for anal in srcmorphanalyzer(phraseEntry[0][0])];
      tgtLemmas = [anal for anal in tgtmorphanalyzer(phraseEntry[0][1])];
    except UnicodeEncodeError:
      print >>sys.stderr, "Unable to handle: %s\t%s" %(phraseEntry[0][0], phraseEntry[0][1]);
      continue;
    
    if len(srcLemmas) == 0:
      print >>sys.stderr, "Abstract entries not available: %s\t%s\t%g" %(phraseEntry[0][0], phraseEntry[0][1], normpmiValue);
    elif len(srcLemmas) != 0 and len(tgtLemmas) == 0:
      srcBestLemma = pruneLemmasList(srcLemmas);
      print >>sys.stderr, "Concrete entry in target missing: %s\t%s\t%s\t%g" %(phraseEntry[0][0], phraseEntry[0][1], srcBestLemma, normpmiValue);
    else:
      srcBestLemma = pruneLemmasList(srcLemmas);
      tgtBestLemma = pruneLemmasList(tgtLemmas);
      print "%s\t%s\t%s\t%s\t%g" %(phraseEntry[0][0], phraseEntry[0][1], srcBestLemma, tgtBestLemma, normpmiValue);

	#if count == 10: break;
  return;

if __name__ == '__main__':
  #main = getPhraseTable;
  main = getLexicon;
  #main = getStatistics;
  #main = getGFLexicon;
  
  try:
    cProfile.run("main(tuple(sys.argv[1:]))", "profiler")
    programStats = pstats.Stats("profiler")
    programStats.sort_stats('cumulative').print_stats()
  except KeyboardInterrupt:
    sys.exit(1)
