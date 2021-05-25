
import json
from argparse import ArgumentParser
import midi
import glob
import copy
import os
import numpy as np
import pretty_midi
from pprint import pprint
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

parser = ArgumentParser()
parser.add_argument('--set1dir', required=True, type=str,
                    help='Path (absolute) to the first dataset (folder)')
parser.add_argument('--set2dir', required=True, type=str,
                    help='Path (absolute) to the second dataset (folder)')
parser.add_argument('--outfile', required=True, type=str,
                    help='File (pickle) where the analysis will be stored')
args = parser.parse_args()

set1 = glob.glob(os.path.join(args.set1dir, '*'))
set2 = glob.glob(os.path.join(args.set2dir, '*'))


num_samples = min(len(set2), len(set1))
evalset = { 
            'total_used_pitch': np.zeros((num_samples, 1))
          # , 'total_used_note': np.zeros((num_samples, 1))
          , 'pitch_range': np.zeros((num_samples, 1))
          # , 'avg_pitch_shift': np.zeros((num_samples, 1))
          , 'avg_IOI': np.zeros((num_samples, 1))
          # , 'total_used_note': np.zeros((num_samples, 1))
          # , 'bar_used_pitch': np.zeros((num_samples, 1, 1))
          # , 'bar_used_note': np.zeros((num_samples, 1))
          , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
          # , 'bar_pitch_class_histogram': np.zeros((num_samples, 1))
          # , 'note_length_hist': np.zeros((num_samples, 12))
          , 'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12))
          # , 'note_length_transition_matrix': np.zeros((num_samples, 12, 12))
          }
metrics_list = evalset.keys()

single_arg_metrics = (
    [ 'total_used_pitch'
    , 'avg_IOI'
    , 'total_pitch_class_histogram'
    , 'bar_pitch_class_histogram'
    , 'pitch_range'
    ])

# Process First Dataset metrics
set1_eval = copy.deepcopy(evalset)
# print('set1')
# pprint(set1[:num_samples])
for i in range(0, num_samples):
    feature = core.extract_feature(set1[i])
    for metric in metrics_list:
        evaluator = getattr(core.metrics(), metric)
        if metric in single_arg_metrics:
            set1_eval[metric][i] = evaluator(feature)
        else:
            set1_eval[metric][i] = evaluator(feature, 0)
# print('metric_set')
# pprint(set1_eval)

# Process First Dataset metrics
set2_eval = copy.deepcopy(evalset)
# print('set2')
# pprint(set2[:num_samples])
for i in range(0, num_samples):
    feature = core.extract_feature(set2[i])
    for metric in metrics_list:
        evaluator = getattr(core.metrics(), metric)
        if metric in single_arg_metrics:
            set2_eval[metric][i] = evaluator(feature)
        else:
            set2_eval[metric][i] = evaluator(feature, 0)
# print('metric_set')
# pprint(set2_eval)

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


# Calculate Intra-set Metrics
for i, metric in enumerate(metrics_list):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][i] = utils.c_dist(
            set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
        set2_intra[test_index[0]][i] = utils.c_dist(
            set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

# Calculate Inter-set Metrics
for i, metric in enumerate(metrics_list):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])


plot_set1_intra = np.transpose(
    set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
plot_set2_intra = np.transpose(
    set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
plot_sets_inter = np.transpose(
    sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)


output = {}
for i, metric in enumerate(metrics_list):
    # print('calculating kl of: {}'.format(metric))

    mean = np.mean(set1_eval[metric], axis=0).tolist()
    std = np.std(set1_eval[metric], axis=0).tolist()

    kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
    ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
    kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
    ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])

    print(kl1)
    print(kl2)
    output[metric] = [mean, std, kl1, ol1, kl2, ol2]


# Save output
if os.path.exists(args.outfile):
    os.remove(args.outfile)

output_file = open(args.outfile, 'w')
json.dump(output, output_file)
output_file.close()

print('Saved output to file: ' + args.outfile)
