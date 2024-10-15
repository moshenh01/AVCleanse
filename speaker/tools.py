'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch, warnings, glob
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

def init_system(args):
    # do not show simpel warnings while running the code, not errors.
    warnings.simplefilter("ignore")

    # training the model with multiple processes cpu and gpu.
    # the " set_sharing_strategy('file_system') " is used to share the memory between the processes.
    # the file_system strategy  ensures that the shared memory between processes is managed via file-based shared memory.
    torch.multiprocessing.set_sharing_strategy('file_system')

    # save the score file in the save path.
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')

    # This line creates a directory path where the model's audio component (model_a) will be saved.
    args.model_save_path_a    = os.path.join(args.save_path, 'model_a')

    # This line creates the directory specified by args.model_save_path_a (which is 'model_a' under args.save_path).
    os.makedirs(args.model_save_path_a, exist_ok = True)

    # This line uses the "glob" library to search for all files
    # that match a certain pattern in the directory model_save_path_a.
    # The pattern 'model_0*.model' looks for files that start with 'model_0' and have a ".model" extension.
    # The result is a list of file paths that match this pattern, and it is stored in args.modelfiles_a.
    # This will allow the system to load or reference previously saved models if needed.
    args.modelfiles_a = glob.glob('%s/model_0*.model'%args.model_save_path_a)

    # This line sorts the list of model files (args.modelfiles_a) in ascending order.
    # as the system might need to load the latest checkpoint model.
    args.modelfiles_a.sort()

    # This line opens the score.txt file in the args.score_save_path in "append" mode ("a+"),
    # meaning new scores will be added to the end of the file, and if the file doesn't exist, it will be created.
    args.score_file = open(args.score_save_path, "a+")
    return args

# The tuneThresholdFromScore function is designed to find optimal decision thresholds for a binary classification system,
# specifically to minimize certain error rates such as "false acceptance rate" (FAR) or "false rejection rate" (FRR).
# It also calculates the Equal Error Rate (EER), which is the point where FAR equals FRR.
# This function is commonly used in verification systems like speaker recognition,
# where determining an appropriate threshold is critical for performance.
def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    # target_fa: A list of target false acceptance rates (FAR). The function will tune the threshold to achieve these target FARs.
    # target_fr: A list of target false rejection rates (FRR). the function will also tune the threshold to achieve these FRRs.

    # fpr (False Positive Rate): The fraction of negative examples that are incorrectly classified as positive.
    # tpr (True Positive Rate): The fraction of positive examples correctly classified as positive.
    # thresholds: The decision thresholds corresponding to the fpr and tpr values.
    # here, for each threshold, the fpr and tpr values are calculated using the labels and scores.
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # The False Negative Rate (FNR) is calculated as 1 - tpr, since FNR is the complement of TPR
    # (i.e., the fraction of positive examples that are incorrectly classified as negative).
    fnr = 1 - tpr

    tunedThreshold = [] # A list to store the tuned thresholds and corresponding FAR and FRR values.

    # Tuning Threshold Based on False Rejection Rates (FRRs):
    # If target_fr is provided, the function tries to find thresholds that match the desired false rejection rates.
    # numpy.nanargmin(numpy.absolute((tfr - fnr))): This line finds the index (idx) of the threshold
    # where the actual false rejection rate (fnr) is closest to the target false rejection rate (tfr).
    # For each tfr, the function appends the corresponding threshold, fpr, and fnr values to the tunedThreshold list.
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    # same here for the target_fa
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    # Equal Error Rate (EER) Calculation:
    # The EER is the error rate at which the false acceptance rate (FAR) equals the false rejection rate (FRR)
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    # The EER is calculated as the maximum of fpr[idxE] and fnr[idxE] and is then multiplied by 100 to express it as a percentage.
    eer  = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr



# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      # labels are sorted based on the indexes of the sorted scores.
      # basically, the labels are sorted in the same order as the scores.
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], correct