from collections import Counter
import random

import numpy as np


class StatLearningExperiment(object):

    def __init__(self, words, size=30):
        self.words = words
        self.create_experiment(size)
        self.create_transisional_probability_matrix()

    @staticmethod
    def remove_word(toremove, words):
        return [word for word in words if word != toremove]

    def create_experiment(self, size):
        stream = []
        counter = Counter(self.words)
        last_word = None
        for i in range(size):
            getmin = min(counter.itervalues())
            possible_words = self.remove_word(
                last_word,
                [x for x, y in counter.iteritems() if y == getmin])
            next_word = random.choice(possible_words)
            stream.append(next_word)
            last_word = next_word
            counter[next_word] += 1

        stream_no_spaces = ''.join(stream)

        self.syllables = [stream_no_spaces[i:i+2]
                          for i in range(0, len(stream_no_spaces), 2)]
        self.syllable_set = list(set(self.syllables))
        self.syllable_dictionary = {
            v: i for i, v in enumerate(self.syllable_set)}
        self.syllable_dictionary_reverse = {
            v: i for i, v in self.syllable_dictionary.iteritems()}

        self.bigrams = [tuple(self.syllables[i:i+2])
                        for i in range(0, len(self.syllables) - 1)]

        self.syllableCounter = Counter(self.syllables)
        self.bigram_counter = Counter(self.bigrams)

        self.probabilities = {tuple([x1, x2]): float(y) /
                              self.syllableCounter[x1]
                              for (x1, x2), y in self.bigram_counter.items()}

    def create_transisional_probability_matrix(self):
        self.transition_matrix = np.zeros([len(self.syllable_set),
                                           len(self.syllable_set)])

        for (w1, w2), p in self.probabilities.iteritems():
            self.transition_matrix[self.syllable_dictionary[w1],
                                   self.syllable_dictionary[w2]] = p

        for i in xrange(self.transition_matrix.shape[0]):
            self.transition_matrix[i] /= self.transition_matrix[i].sum()

        np.savetxt('transition_matrix.txt', self.transition_matrix)
        with file('word_order.txt', 'w') as fout:
            fout.write(' '.join(sorted(self.syllable_dictionary,
                                       key=self.syllable_dictionary.get)))

    def get_random_word(self, nb_syllables=3):
        ans = [random.choice(self.syllable_set)]
        probs = []
        for i in range(nb_syllables - 1):
            pi = self.syllable_dictionary[ans[-1]]
            next_id = int(np.random.choice(
                self.transition_matrix.shape[0], 1,
                p=self.transition_matrix[pi]))
            ans.append(self.syllable_dictionary_reverse[next_id])
            probs.append(self.transition_matrix[pi, next_id])
        ans = ' '.join(ans)
        probs = np.array(probs).prod()
        return (ans, probs)

    def save_to_file(self, filelocation='words.txt'):
        with open(filelocation, 'w') as fout:
            fout.write(' '.join(self.syllables))

    def __str__(self):
        ans = '\n'
        for (x1, x2), v in self.probabilities.iteritems():
            ans += "{} --> {}:\t{}\n".format(x1, x2, v)
        return ans


def determine_best(words, size=30, nb_simulations=10):
    return min([StatLearningExperiment(words, size)
                for i in xrange(nb_simulations)],
               key=lambda x: np.array(x.probabilities.values()).std())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str,
                        help="The file containing space-separated words")
    parser.add_argument("-s", "--size", type=int,
                        help="How many true words will be there.")
    parser.add_argument("-r", "--repetitions", type=int,
                        help="How many times to repeat until the best one \
is found.")
    parser.add_argument("-g", "--generate", type=int,
                        default=0, help="Number of words to generate")
    args = parser.parse_args()

    with open(args.file, "r") as fin:
        words = fin.read().strip().split()

    sle = determine_best(words, args.size, args.repetitions)
    if args.generate:
        results = dict([sle.get_random_word() for i in range(args.generate)])
        sort_res = sorted(results, key=results.get, reverse=True)
        for word in sort_res:
            print '{}\t{}'.format(word, results[word])
    sle.save_to_file()
