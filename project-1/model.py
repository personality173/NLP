import math
import numpy as np

class Model:
    def __init__(self, alpha=1):
        self.vocab = set()
        self.spam = {}
        self.ham = {}
        self.alpha = alpha

        self.label2num = None
        self.num2label = None

        self.Nvoc = 0
        self.Nspam = 0
        self.Nham = 0

        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

        self.p_spam_prior = 0.5
        self.p_ham_prior = 0.5

    def fit(self, dataset):
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test

        self.label2num = dataset.label2num
        self.num2label = dataset.num2label

        # сброс
        self.vocab = set()
        self.spam = {}
        self.ham = {}

        # priors
        spam_id = self.label2num.get("spam", None)
        if spam_id is None:
            spam_id = 1  # на всякий случай
        n_spam = int(np.sum(self._train_y == spam_id))
        n_total = len(self._train_y)
        self.p_spam_prior = n_spam / n_total
        self.p_ham_prior = 1 - self.p_spam_prior

        # word counts
        for words, y in zip(self._train_X, self._train_y):
            label = self.num2label[y]
            target = self.spam if label == "spam" else self.ham

            for w in words:
                self.vocab.add(w)
                target[w] = target.get(w, 0) + 1

        self.Nvoc = len(self.vocab)
        self.Nspam = sum(self.spam.values())
        self.Nham = sum(self.ham.values())

    def inference(self, message):
        # log priors
        pspam = math.log(self.p_spam_prior + 1e-12)
        pham  = math.log(self.p_ham_prior + 1e-12)

        denom_spam = self.Nspam + self.alpha * self.Nvoc
        denom_ham  = self.Nham + self.alpha * self.Nvoc

        for w in message:
            pspam += math.log((self.spam.get(w, 0) + self.alpha) / denom_spam)
            pham  += math.log((self.ham.get(w, 0) + self.alpha) / denom_ham)

        return "spam" if pspam > pham else "ham"

    def validation(self):
        correct = 0
        total = len(self._val_y)

        for x, y in zip(self._val_X, self._val_y):
            pred = self.inference(x)
            if pred == self.num2label[y]:
                correct += 1

        return correct / total if total > 0 else 0.0

    def test(self):
        correct = 0
        total = len(self._test_y)

        for x, y in zip(self._test_X, self._test_y):
            pred = self.inference(x)
            if pred == self.num2label[y]:
                correct += 1

        return correct / total if total > 0 else 0.0



