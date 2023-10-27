from models.pot import SPOT
import numpy as np

def get_t(init_score, score, q=1e-5):
    lm = (0.99999, 1)
    lms = lm[0]
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except: lms = lms * 0.999
        else: break

    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm[1]
    return pot_th
