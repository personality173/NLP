import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X  # сообщения
        self._y = y  # метки ["spam", "ham"]
        self.train = None
        self.val = None
        self.test = None
        self.label2num = {}
        self.num2label = {}
        self._transform()

    def __len__(self):
        return len(self._x)

    def _clean_text(self, text: str):
        text = text.lower()
        # оставляем буквы+цифры, остальное -> пробел
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()

        # биграммы
        bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _transform(self):
        # метки -> числа (фиксируем порядок, чтобы было стабильно)
        labels = sorted(list(set(self._y)))
        self.label2num = {label: i for i, label in enumerate(labels)}
        self.num2label = {i: label for label, i in self.label2num.items()}

        # чистка и токенизация
        self._x = [self._clean_text(x) for x in self._x]
        self._y = np.array([self.label2num[y] for y in self._y])

    def split_dataset(self, val=0.1, test=0.1):
        n = len(self._x)
        idx = np.random.permutation(n)

        test_size = int(n * test)
        val_size = int(n * val)

        test_idx = idx[:test_size]
        val_idx = idx[test_size:test_size + val_size]
        train_idx = idx[test_size + val_size:]

        X = np.array(self._x, dtype=object)
        y = self._y

        self.train = (X[train_idx], y[train_idx])
        self.val = (X[val_idx], y[val_idx])
        self.test = (X[test_idx], y[test_idx])

