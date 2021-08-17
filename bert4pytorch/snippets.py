#! -*- coding: utf-8 -*-
# 其他代码合集

import numpy as np

def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences