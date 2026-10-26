from collections import defaultdict

import numpy as np


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        # map each token to every token that follows it; duplicates are kept
        # so that random sampling reflects the actual transition frequencies
        tokens = self.corpus.split()

        term_dict = defaultdict(list)
        for current, following in zip(tokens[:-1], tokens[1:]):
            term_dict[current].append(following)

        self.term_dict = dict(term_dict)

        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        # generate term_count tokens, starting from seed_term or a random token
        if self.term_dict is None:
            self.get_term_dict()

        keys = list(self.term_dict.keys())

        if seed_term is None:
            current = np.random.choice(keys)
        elif seed_term in self.term_dict:
            current = seed_term
        else:
            raise ValueError(f"Seed term '{seed_term}' not found in corpus.")

        words = [current]
        while len(words) < term_count:
            # a token that only appears at the very end of the corpus has no
            # followers, so jump to a random token to keep the chain going
            followers = self.term_dict.get(current)
            if not followers:
                current = np.random.choice(keys)
            else:
                current = np.random.choice(followers)
            words.append(current)

        return ' '.join(words)
