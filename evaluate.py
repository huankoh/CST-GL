import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm
from multiprocessing import Pool

## EVALUATING FUNCTION
def pointwise_evaluation(gt_labels, pred_labels, scoring):

    ## non-iterative
    auto_recall = recall_score(gt_labels, pred_labels)
    auto_precision = precision_score(gt_labels, pred_labels)
    auto_f1 = f1_score(gt_labels, pred_labels)
    roc = roc_auc_score(gt_labels, scoring)
    prc = average_precision_score(gt_labels, scoring)

    scoring = np.array(scoring)
    sorted_scoring = np.sort(np.array(scoring))

    best_recall, best_precision, best_f1 = 0, 0, 0
    ## iteratively to get best f1 result
    pointwise_f1_evaluator = pointwise_best_f1(gt_labels,scoring)
    f1s = pointwise_f1_evaluator.go()

    best_index = f1s.index(max(f1s))
    best_thres = sorted_scoring[best_index]
    best_pred = scoring > best_thres
    # Best results
    best_f1 = f1_score(gt_labels,best_pred)
    best_recall = recall_score(gt_labels,best_pred)
    best_precision = precision_score(gt_labels,best_pred)

    output = dict(auto_precision=auto_precision,auto_recall=auto_recall,auto_f1=auto_f1,
                  roc=roc,prc=prc,best_precision=best_precision,
                  best_recall=best_recall,best_f1=best_f1)
    return output


def early_detection_evaluation(truth, scoring, delay=[0,6,30,60,120,180,360]):

    assert len(truth) == len(scoring)

    output = {}

    for d in delay:
        early_evaluator = early_best_f1(truth,scoring,d)
        early_results = early_evaluator.go()
        best_score = max(early_results)
        output["delay_"+str(d)] = best_score

    return output



class pointwise_best_f1():
   def __init__(self,gt_labels,scoring):
       self.gt_labels = gt_labels
       self.scoring = scoring
       self.sorted_scoring = np.sort(np.array(scoring))

   def f1_with_thres(self,index):
       thres = self.sorted_scoring[index]
       pred_labels = self.scoring > thres
       f1 = f1_score(self.gt_labels, pred_labels)
       return f1

   def go(self):

      print('processing pointwise best f1')
      p = Pool()
      sc = p.map(self, range(len(self.scoring)))

      return sc

   def __call__(self, x):
      return self.f1_with_thres(x)


class early_best_f1():
    def __init__(self, gt_labels, scoring,delay):
        self.gt_labels = gt_labels
        self.scoring = scoring
        self.sorted_scoring = np.sort(np.array(scoring))
        self.delay = delay

    # Multi pool
    def early_f1(self,idx):
        threshold = self.sorted_scoring[idx]
        result = np.array(self.scoring) > threshold
        score = label_evaluation(self.gt_labels, result.tolist(), self.delay)
        return score

    def go(self):
        print('processing early detection best f1 with delay ',str(self.delay))
        p = Pool()
        sc = p.map(self, range(len(self.scoring)))
        return sc

    def __call__(self, x):
        return self.early_f1(x)


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=int)
    new_label[idx] = label

    return new_label


def label_evaluation(truth_list, result_list, delay=7):
    truth_df = {'timestamp': range(len(truth_list)), 'label': truth_list}
    result_df = {'timestamp': range(len(result_list)), 'predict': result_list}

    y_true_list = []
    y_pred_list = []
    # Adapted from: https://arxiv.org/pdf/1906.03821.pdf
    truth = truth_df
    y_true = reconstruct_label(truth["timestamp"], truth["label"])
    result = result_df
    y_pred = reconstruct_label(result["timestamp"], result["predict"])
    y_pred = get_range_proba(y_pred, y_true, delay)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    # run f1score
    fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    return fscore

