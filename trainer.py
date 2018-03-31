import json
import sys
from operator import itemgetter
from copy import deepcopy

from tableDistance import distance

VERBOSE = False


def getCorpus(filename):
    """
        Load corpus file located at `filename` into a list of dicts
    """
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    if VERBOSE:
        print(corpus, file=sys.stderr)
    return corpus


def getWords(corpus):
    """
    From a `corpus` object, build a dict whose keys are 'en' and 'fr',
    and whose values are sets. Each dict[language] set contains every
    word in that language which appears in the corpus
    """

    def sourceWords(lang):
        for pair in corpus:
            for word in pair[lang].split():
                yield word

    return {lang: set(sourceWords(lang)) for lang in ('A', 'B')}


def initTranslationProbabilities(corpus):
    """
    Given a `corpus` generate the first set of translation probabilities,
    which can be accessed as p(e|s) <=> translation_probabilities[e][s]
    we first assume that for an `e` and set of `s`s, it is equally likely
    that e will translate to any s in `s`s
    """
    words = getWords(corpus)
    return {
        wordEn: {wordFr: 1 / len(words['A'])
                  for wordFr in words['B']}
        for wordEn in words['A']}


def trainIteration(corpus, words, totals, prevTranslationProbabilities):
    """
    Perform one iteration of the EM-Algorithm

    corpus: corpus object to train from
    words: {language: {word}} mapping

    total_s: counts of the destination words, weighted according to
             their translation probabilities t(e|s)

    prev_translation_probabilities: the translation_probabilities from the
                                    last iteration of the EM algorithm
    """
    translationProbabilities = deepcopy(prevTranslationProbabilities)

    counts = {wordEn: {wordFr: 0 for wordFr in words['B']}
              for wordEn in words['A']}

    totals = {wordFr: 0 for wordFr in words['B']}

    for (es, fs) in [(pair['A'].split(), pair['B'].split())
                     for pair in corpus]:
        for e in es:
            totals[e] = 0

            for f in fs:
                totals[e] += translationProbabilities[e][f]

        for e in es:
            for f in fs:
                counts[e][f] += (translationProbabilities[e][f] /
                                 totals[e])
                totals[f] += translationProbabilities[e][f] / totals[e]

    for f in words['B']:
        for e in words['A']:
            translationProbabilities[e][f] = counts[e][f] / totals[f]

    return translationProbabilities


def isConverged(probabiltiesPrev, probabiltiesCurr, epsilon):
    """
    Decide when the model whose final two iterations are
    `probabiltiesPrev` and `probabiltiesCurr` has converged
    """
    delta = distance(probabiltiesPrev, probabiltiesCurr)
    if VERBOSE:
        print(delta, file=sys.stderr)

    return delta < epsilon


def trainModel(corpus, epsilon):
    # Given a `corpus` and `epsilon`, train a translation model on that corpus

    words = getWords(corpus)

    totals = {wordEn: 0 for wordEn in words['A']}
    prevTranslationProbabilities = initTranslationProbabilities(corpus)

    converged = False
    iterations = 0
    while not converged:
        translationProbabilities = trainIteration(
            corpus, words, totals,
            prevTranslationProbabilities
        )

        converged = isConverged(prevTranslationProbabilities,
                                translationProbabilities, epsilon)
        prevTranslationProbabilities = translationProbabilities
        iterations += 1
        #print("iteration no: " + str(iterations) + " prev: " + str(prevTranslationProbabilities))
    return translationProbabilities, iterations


def summarizeResults(translationProbabilities):
    """
    from a dict of source: {target: p(source|target}, return
    a list of mappings from source words to the most probable target word
    """
    return {
        # for each english word
        # sort the words it could translate to; most probable first
        k: sorted(v.items(), key=itemgetter(1), reverse=True)
            # then grab the head of that == `(most_probable, p(k|most probable)`
        [0]
            # and the first of that pair (the actual word!)
        [0]
        for (k, v) in translationProbabilities.items()
    }

def main(infile, *, outfile: 'o' = None, epsilon: 'e' = 0.0001, verbose: 'v' = False):
    """
    IBM Model 1 SMT Training Example

    infile: path to JSON file containing English-Bangla sentence pairs
            in the form [ {"en": <sentence>, "bn": <sentence>}, ... ]

    outfile: path to output file (defaults to stdout)

    epsilon: Acceptable euclidean distance between translation probability
             vectors across iterations

    verbose: print running info to stderr
    """

    global VERBOSE
    VERBOSE = verbose

    if infile == '-':
        corpus = getCorpus(sys.stdin)
    else:
        corpus = getCorpus(infile)

    probabilities, iterations = trainModel(corpus, epsilon)
    result_table = summarizeResults(probabilities)
    if outfile:
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(result_table, f, ensure_ascii=False)
    else:
        json.dump(result_table, sys.stdout)

    if VERBOSE:
        print('Performed {} iterations'.format(iterations), file=sys.stderr)




