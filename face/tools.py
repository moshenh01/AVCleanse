'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch, warnings, glob
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

# initializes the system, sets up the save path, and creates the model save path
# also sets up the score save path and the model files
def init_system(args):
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')
    args.model_save_path_v    = os.path.join(args.save_path, 'model_v')
    os.makedirs(args.model_save_path_v, exist_ok = True)
    args.modelfiles_v = glob.glob('%s/model_0*.model'%args.model_save_path_v)
    args.modelfiles_v.sort()
    args.score_file = open(args.score_save_path, "a+")
    return args


# Tunes the decision threshold to optimize for a target false alarm or false reject rate
# returning the optimal threshold and EER.
# Thresholds are used to determine whether a given score is a match or not.
# The EER is the point where the false accept rate and false reject rate are equal.
def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = []
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
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
# The MinDcf is the minimum of the detection cost function.
# The detection cost function is a weighted sum of the false negative, means the system fails to detect the target speaker, and false positive, means the system detects the target speaker when it is not.
# So the MinDCF is used to evaluate the performance of a speaker verification system.
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

#  Computes the accuracy of the model
#  The accuracy is the number of correct predictions divided by the total number of predictions
def accuracy(output, target, topk=(1,)):

	maxk = max(topk) # topk is the number of top-k predictions to check
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res[0], correct