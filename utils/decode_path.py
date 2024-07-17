import numpy as np


def decode_best_path(probs, bl=0):
    seq_list = []
    for pr in probs:
        best_path = np.argmax(pr, axis=0).tolist()
        seq = []
        for i, b in enumerate(best_path):
            if b == bl:
                continue
            elif i != 0 and b == best_path[i - 1]:
                continue
            else:
                seq.append(b)
        seq_list.append(seq)
    return seq_list