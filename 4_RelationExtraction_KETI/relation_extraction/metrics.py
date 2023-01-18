
import numpy as np
import sklearn.metrics


def f1_score_micro_sample_weight_dict(targets, predictions, ex_num=-1):
    sample_weight = np.array(targets != ex_num, dtype=np.float64)
    return {"f1_score":
            {
                'score': 100*sklearn.metrics.f1_score(
                    targets, predictions, average='micro', sample_weight=sample_weight, zero_division=0),
                'count': np.sum(sample_weight)
            }
            }
