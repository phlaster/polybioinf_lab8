import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, kstest
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from argparse import ArgumentParser
from tqdm import tqdm
from joblib import Parallel, delayed


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import utils


def parse_args():
    parser = ArgumentParser(description='Lab 8')
    parser.add_argument('--bed_ref', required=True, help='Path to reference BED file.')
    parser.add_argument('--bed_predicted', required=True, help='Path to predicted BED file.')
    parser.add_argument('--hist', required=True, help='Path to histogram output')
    parser.add_argument('--cpu', type=int, default=1, help='Number of CPU cores to use.')
    return parser.parse_args()


# def calculate_intersections(ground_truth, predictions, n_cpu=-1):
#     def calculate_for_prediction(pred):
#         chrom_matches = ground_truth[ground_truth['chromosome'] == pred['chromosome']]
#         intersections = [
#             max(0, min(truth['end'], pred['end']) - max(truth['start'], pred['start']))
#             for _, truth in chrom_matches.iterrows()
#             if max(truth['start'], pred['start']) < min(truth['end'], pred['end'])
#         ]
#         return intersections

#     results = Parallel(n_jobs=n_cpu)(delayed(calculate_for_prediction)(pred) for _, pred in predictions.iterrows())
#     intersections = [length for sublist in results for length in sublist]
#     return intersections


def plot_histogram_and_fit(intersections, filename):
    plt.hist(intersections, bins=500, alpha=0.7, color='blue', density=True, label='Histogram')
    
    mean, std = norm.fit(intersections)
    loc, scale = cauchy.fit(intersections)
    x = np.linspace(min(intersections), max(intersections), 1000)
    
    pdf_norm = norm.pdf(x, mean, std)
    
    pdf_cauchy = cauchy.pdf(x, loc, scale)
    
    plt.plot(x, pdf_norm, 'r-', label=f'Normal Fit: mean={mean:.2f}, std={std:.2f}')
    plt.plot(x, pdf_cauchy, 'b--', label=f'Cauchy Fit: loc={loc:.2f}, scale={scale:.2f}')
    
    plt.xlabel('Intersection Length')
    plt.ylabel('Density')
    plt.title('Histogram and distributions fits')
    plt.legend()
    plt.xlim(0, 1500)
    plt.savefig(filename)
    plt.close()
    return mean, std, loc, scale


def ks_cauchy_test(loc, scale, intersections):
    statistic, p_value = kstest(intersections, 'cauchy', args=(loc, scale))
    return p_value

def ks_cauchy_test(mean, std, intersections):
    statistic, p_value = kstest(intersections, 'norm', args=(mean, std))
    return p_value

def intersections_and_metrics(ground_truth, predictions, n_cpu=-1):
    def process_prediction(pred):
        chrom_matches = ground_truth[ground_truth['chromosome'] == pred['chromosome']]
        intersections = []
        for _, truth in chrom_matches.iterrows():
            if max(truth['start'], pred['start']) < min(truth['end'], pred['end']):
                intersection_length = max(0, min(truth['end'], pred['end']) - max(truth['start'], pred['start']))
                intersections.append(intersection_length)
        flag = 1 if intersections else 0
        return intersections, flag

    results = Parallel(n_jobs=n_cpu)(delayed(process_prediction)(pred) for _, pred in predictions.iterrows())
    intersections = [length for sublist in [res[0] for res in results] for length in sublist]
    y_true_flags = [res[1] for res in results]
    y_pred = [1] * len(predictions)
    padding = len(ground_truth) - len(predictions)
    y_true = y_true_flags + [0] * padding
    y_pred = y_pred + [0] * padding
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    sensitivity = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)

    return intersections, specificity, sensitivity, f_score, tn, fp, fn, tp


# Основной скрипт
if __name__ == "__main__":
    args = parse_args()

    bed_ref = args.bed_ref
    bed_predicted = args.bed_predicted
    cpu = args.cpu
    hist = args.hist

    ground_truth = utils.read_bed(bed_ref)
    predictions = utils.read_bed(bed_predicted)
    intersections, specificity, sensitivity, f_score, tn, fp, fn, tp = intersections_and_metrics(ground_truth, predictions, n_cpu=cpu)
    mean, std, loc, scale = plot_histogram_and_fit(intersections, hist)
    norm_sign = ks_cauchy_test(mean, std, intersections)
    kauchy_sign = ks_cauchy_test(loc, scale, intersections)
    

    print(f"Intersections calculated: {len(intersections)}")
    print(f"Normal distributuion: {'Significant' if norm_sign < 0.05 else 'Not Significant'}, p-value={norm_sign:.2e}")
    print(f"Kauchy distributuion: {'Significant' if kauchy_sign < 0.05 else 'Not Significant'}, p-value={kauchy_sign:.2e}")
    print("Confusion Matrix:")
    print(f"\tTRUE\tFALSE\nPOS\t{tp}\t{fp}\nNEG\t{tn}\t{fn}")
    print(f"Specificity: {specificity:.2f}, Sensitivity: {sensitivity:.2f}, F-score: {f_score:.2f}")
