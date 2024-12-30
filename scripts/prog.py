import sys, os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import utils

# Very important!
SEED=8

def parse_args():
    parser = ArgumentParser(description='Lab 8')
    parser.add_argument('--fasta', required=True, help='Path to the fasta file.')
    parser.add_argument('--bed_in', required=True, help='Path to the BED file.')
    parser.add_argument('--bed_out', required=True, help='Path to the output BED file.')
    parser.add_argument('--model_out', required=True, help='Path to the output model parameters JSON file.')
    parser.add_argument('--n_chroms', type=int, default=9999, help='Number of sequences to read.')
    parser.add_argument('--cpu', type=int, default=1, help='Number of CPU cores to use.')
    return parser.parse_args()

def _predict_sequence(model, sequence):
    return model.predict(sequence.reshape(-1, 1))

def parallel_predict(model, fasta, max_workers=None):
    predicted = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(_predict_sequence, model, sequence): chrom for chrom, sequence in fasta.items()}
        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Predicting", leave=False):
            chrom = future_to_key[future]
            predicted[chrom] = future.result()
    return predicted

def average_hmm_models(models: list, weights=None) -> hmm.CategoricalHMM:
    def _normalize(probs, axis=None):
        return probs / np.sum(probs, axis=axis, keepdims=True)
    
    # Weighted average for different types of parameters
    def _weighted_average_1d(params_list):
        # For one-dimensional parameters (startprob_)
        params_array = np.array([np.exp(params) for params in params_list])
        return np.sum(params_array * weights[:, np.newaxis], axis=0)
    
    def _weighted_average_2d(params_list):
        # For two-dimensional parameters (transmat_, emissionprob_)
        params_array = np.array([np.exp(params) for params in params_list])
        return np.sum(params_array * weights[:, np.newaxis, np.newaxis], axis=0)
    
    def _check_probabilities(probs, name):
        sums = np.sum(probs, axis=1)
        if not np.allclose(sums, 1.0, rtol=1e-5):
            raise ValueError(f"Probabilities in {name} do not sum up to 1: {sums}")

    if not models or not all(isinstance(m, hmm.CategoricalHMM) for m in models):
        raise ValueError("Input list has to contain CategoricalHMM models only")
    
    # Checking and normalizing weights
    if weights is not None:
        if len(weights) != len(models):
            raise ValueError("Number of weights must match the number of models")
        if not all(w > 0 for w in weights):
            raise ValueError("All weights must be positive")
        # Normalize weights
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(models)) / len(models)
    
    # Checking the structure of models
    n_components = models[0].n_components
    n_features = models[0].n_features
    
    if not all(m.n_components == n_components and m.n_features == n_features for m in models):
        raise ValueError("All models must have the same number of states and features")
    
    # Averaging
    avg_startprob = _weighted_average_1d([m.startprob_ for m in models])
    avg_startprob = _normalize(avg_startprob)
    
    avg_transmat = _weighted_average_2d([m.transmat_ for m in models])
    avg_transmat = _normalize(avg_transmat, axis=1)
    
    avg_emissionprob = _weighted_average_2d([m.emissionprob_ for m in models])
    avg_emissionprob = _normalize(avg_emissionprob, axis=1)
    
    if not np.allclose(np.sum(avg_startprob), 1.0, rtol=1e-5):
        raise ValueError(f"Sum of initial probabilities is not equal to 1: {np.sum(avg_startprob)}")    
    _check_probabilities(avg_transmat, "transition matrix")
    _check_probabilities(avg_emissionprob, "emission matrix")
    avg_model = hmm.CategoricalHMM(n_components=n_components)
    avg_model.startprob_ = avg_startprob
    avg_model.transmat_ = avg_transmat
    avg_model.emissionprob_ = avg_emissionprob
    return avg_model


def fit_model(chunk):
    global SEED
    model = hmm.CategoricalHMM(n_components=2, random_state=SEED)
    model.fit(chunk.reshape(-1, 1))
    return model

def split_training(chromosome, n_threads=1):    
    chunks = np.array_split(chromosome, n_threads)
    models = []
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        models = list(executor.map(fit_model, chunks))
    chunks_lengths = list(map(len, chunks))
    return average_hmm_models(models, chunks_lengths)

def save_model_to_json(model, filename):
    model_params = {
        "n_components": model.n_components,
        "transmat_": model.transmat_.tolist(),
        "emissionprob_": model.emissionprob_.tolist(),
        "startprob_": model.startprob_.tolist()
    }

    with open(filename, "w") as f:
        json.dump(model_params, f, indent=4)

def load_model_from_json(filename):
    with open(filename, "r") as f:
        model_params = json.load(f)
    model = hmm.CategoricalHMM(n_components=model_params["n_components"])
    model.transmat_ = np.array(model_params["transmat_"])
    model.emissionprob_ = np.array(model_params["emissionprob_"])
    model.startprob_ = np.array(model_params["startprob_"])
    return model

def main():
    start_time = time.time()

    args = parse_args()
    
    utils.cout("[ 1/8 ] Reading input bed file", "blue")
    bed_df = utils.read_bed(args.bed_in)
    
    utils.cout("[ 2/8 ] Mapping from chromosome names to np.arrays", "blue")
    fasta = utils.sequences_to_arrays_parallel(utils.read_fasta(args.fasta, args.n_chroms), args.cpu)
    
    utils.cout("[ 3/8 ] Calculating data masks", "blue")
    genome_masks = utils.mask_regions(fasta, bed_df)
    
    utils.cout("[ 4/8 ] Training", "blue")
    chrom_models = []
    chrom_lengths = []
    for chrom, sequence in tqdm(fasta.items(), desc="Split-training over chromosomes", leave=False):
        chrom_models.append(split_training(sequence, args.cpu))
        chrom_lengths.append(len(sequence))

    utils.cout("[ 5/8 ] Averaging the parameters", "blue")
    aggregated_model = average_hmm_models(chrom_models, chrom_lengths)

    utils.cout("[ 6/8 ] Model inference", "blue")
    predicted = parallel_predict(aggregated_model, fasta, args.cpu)
    
    utils.cout("[ 7/8 ] Writing result to bed file", "blue")
    utils.write_bed(predicted, args.bed_out)

    utils.cout("[ 8/8 ] Saving model to json", "blue")
    save_model_to_json(aggregated_model, args.model_out)

    end_time = time.time()
    total_time = end_time - start_time
    utils.cout(f"All done in {total_time:.1f} s.", "green")

if __name__ == '__main__':
    main()