import random
import re
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.next_words = defaultdict(list)

        for sentence in corpus:
            words = re.findall(r"\b\w+\b", sentence.lower())
            for current, following in zip(words, words[1:]):
                self.next_words[current].append(following)

    def generate_text(self, start_word, length):
        if length < 1:
            return ""

        words = [start_word.strip().lower()]

        for _ in range(length - 1):
            choices = self.next_words.get(words[-1])
            if not choices:
                break
            words.append(random.choice(choices))

        return " ".join(words)
