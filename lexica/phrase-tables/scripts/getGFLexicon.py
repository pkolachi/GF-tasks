
import codecs, itertools, math, multiprocessing, re, string, sys;
from operator import itemgetter;

sys.stdin  = codecs.getreader('utf-8')(sys.stdin);
sys.stdout = codecs.getwriter('utf-8')(sys.stdout);

GOOGLE_POSCATS = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.'];
poscat_MAP = {
	'A'      : ('ADJ',),
	'A2'     : ('ADJ',),
	'AdA'    : ('ADJ',),
	'AdN'    : ('ADJ',),
	'AdV'    : ('ADV',),
	'Adv'    : ('ADV',),
	'Conj'   : ('CONJ',),
	'Det'    : ('DET',),
	'IAdv'   : ('ADV',),
	'IDet'   : ('DET',),
	'IP'     : ('X',), 
	'IQuant' : ('X',),
	'Interj' : ('X',), 
	'N'      : ('NOUN',),
	'N2'     : ('NOUN',),
	'NP'     : ('NOUN',),
	'Num'    : ('NUM',),
	'PConj'  : ('CONJ',),
	'PN'     : ('NOUN', 'name'),
	'Predet' : ('DET',),
	'Prep'   : ('ADP', 'preposition'),
	'Pron'   : ('PRON',),
	'Quant'  : ('NUM',),
	'Subj'   : ('NOUN',),
	'V'      : ('VERB',),
	'V2'     : ('VERB',),
	'V2A'    : ('VERB',),
	'V2Q'    : ('VERB',),
	'V2S'    : ('VERB',),
	'V2V'    : ('VERB',),
	'V3'     : ('VERB',),
	'VA'     : ('VERB',),
	'VQ'     : ('VERB',),
	'VS'     : ('VERB',),
	'VV'     : ('VERB',)
	}

def readDictTable(dictfile):
  dictionary = {};
  for line in random_utils.lines_from_file(dictfile):
    fields = line.split('\t');
    cat, subcat, srclemma, transvariants, sensehint = fields;
    dictionary.setdefault((cat, subcat), {}).setdefault(srclemma, []).append( (transvariants, sensehint) );
  return dictionary;

def getGFAbstractNames(dictionary, pgfConcreteGrammar):
  global poscat_MAP;
  for category in dictionary.keys():
    for lemma in dictionary[category]:
      # get abstract function for the specific lemma; 
      possibleEntries = pgfConcreteGrammar.lookupMorpho(lemma.encode('latin-1', 'ignore'));
      # end up with many options for possible entries; 
      # prune them based on category first; 
      prunedEntries = [];
      for entry in possibleEntries:
        gflemma, gfcategory = entry[0].rsplit('_', 1) if entry[0].find('_') != -1 else ('', entry[0]);
        print poscat_MAP.get(gfcategory, 'NONE'), category, lemma, entry[0];
	  if len(possibleEntries) == 0:
        print 'NOT FOUND', category, lemma;
  return;

def readGFEntries(iterable):
  entry = '';
  for line in iterable:
    lemma, category = line.strip().rsplit('_', 1);
    tokens = lemma.split('_');
    lemma = '_'.join(tokens[:-1]) if tokens[-1] in string.digits else '_'.join(tokens);
    funcname = '%s_%s' %(lemma, category);
    if funcname != entry:
      yield funcname;
      entry = funcname;
      dummy = (yield);

def levenshtein_distance(string1, string2):
  '''	Taken from http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python '''
  if len(string1) < len(string2):
    return levenshtein_distance(string2, string1);
  #print string1, string2;
  if len(string2) == 0:
    return len(string1);
  previous_row = xrange(len(string2)+1);
  for i,ch1 in enumerate(string1):
    current_row = [i+1];
    for j,ch2 in enumerate(string2):
      insertions    = previous_row[j+1] + 1; # j+1 instead of j since previous_row and current_row are one character longer
      deletions     = current_row[j]    + 1; # than s2
      substitutions = previous_row[j]   + (ch1 != ch2);
    current_row.append( min(insertions, deletions, substitutions) );
	previous_row = current_row;
  return float(previous_row[-1])/len(string1);

def levenshtein_distance_multi(args):
  return levenshtein_distance(*args);

def argmin(pairsList):
  return min(pairsList, key=itemgetter(1))[0];

def findExactEntryforAbstractFunction(abstractFuncName, dictionary=None, gfGrammar=None):
  if gfGrammar == None:
    return None;

def findClosestEntrytoAbstractFunction(abstractFuncName, dictionary=None, pool=None):
  global poscat_MAP;
  if dictionary == None:
    return None;
  lemma, gcategory = abstractFuncName.rsplit('_', 1);
  common_category = poscat_MAP.get(gcategory);
  tokens = lemma.split('_');
  lemma = ' '.join(tokens[:-1]) if tokens[-1] in string.digits else ' '.join(tokens);
  N, chunksize = 10000, 1000;
  lemmasList, categoriesList, stringDistList = [], [], [];
  for category in filter(lambda X: X[0] == common_category[0], dictionary.keys()):
    #print >>sys.stderr, "given category=%r\tlooking for category=%r\tfound=%r" %(gcategory, common_category, category);
    lemmasList.extend( dictionary[category].keys() );
    categoriesList.extend( itertools.repeat(category, len(dictionary[category].keys())) );
  if len(lemmasList) == 0 or lemmasList == None:
    return None;
  zipList = itertools.izip( itertools.imap(unicode.lower, lemmasList), itertools.repeat(lemma, len(lemmasList)) );
  if pool == None:
    stringDistList = itertools.imap(levenshtein_distance_multi, zipList);
  else:
    while True:
      min_result = pool.map(levenshtein_distance_multi, itertools.islice(zipList, N), chunksize=chunksize);
      if min_result:
        stringDistList.extend(min_result);
      else:
		break;
  maxIdx = argmin(enumerate(stringDistList));
  return (lemmasList[maxIdx], categoriesList[maxIdx], dictionary[categoriesList[maxIdx]][lemmasList[maxIdx]]);

def main():
  dictionary = readDictTable(sys.argv[1]);
  if len(sys.argv) > 2:
    import pgf;
    gfGrammar, langname = pgf.readPGF(sys.argv[2]), sys.argv[3];
    dictEntries_GFmap = getGFAbstractNames(dictionary, gfGrammar.languages[langname]);
  else:
    gfGrammar = None;
    
  gf_entries = readGFEntries(sys.stdin);
  pool = multiprocessing.Pool(3);
  for line in gf_entries:
    #entry = findExactEntryforAbstractFunction(line.strip(), dictionary, gfGrammar);
    entry = findClosestEntrytoAbstractFunction(line.strip(), dictionary, pool=pool);
    if entry == None:
      print '%s\tNone' %(line.strip());
    else:
      matchedLemma = entry[0];
      lemmaCategory = '%s_%s' %(entry[1][0], entry[1][1]);
      lemmaCategory = lemmaCategory.strip('_');
      for idx, sense in enumerate(entry[2]):
        lemma, category = line.strip().rsplit('_', 1);
        newname = '%s_%d_%s' %(lemma, idx+1, category) if len(entry[2]) > 1 else line;
        print '%s\t%s\t%s\t%s\t%s\t%s' %(line, newname, matchedLemma, lemmaCategory, sense[0], sense[1]);
	gf_entries.send('finished');
  return;

if __name__ == '__main__':
  main();
