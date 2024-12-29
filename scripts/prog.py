import json
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from tqdm import tqdm


import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import utils

def _predict_sequence(model, sequence):
    return model.predict(sequence.reshape(-1, 1))

def parallel_predict(model, fasta, max_workers=None):
    predicted = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(_predict_sequence, model, sequence): chrom for chrom, sequence in fasta.items()}

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Predicting"):
            chrom = future_to_key[future]
            predicted[chrom] = future.result()

    return predicted

def average_hmm_models(models: list, weights=None) -> hmm.CategoricalHMM:
    """
    Усредняет параметры нескольких HMM моделей с учетом весов.
    
    Args:
        models: список моделей CategoricalHMM
        weights: список весов для каждой модели (например, длины обучающих последовательностей)
                если None, используется равное взвешивание
    """
    def _normalize(probs, axis=None):
        return probs / np.sum(probs, axis=axis, keepdims=True)
    
    # Взвешенное усреднение для разных типов параметров
    def _weighted_average_1d(params_list):
        # Для одномерных параметров (startprob_)
        params_array = np.array([np.exp(params) for params in params_list])
        return np.sum(params_array * weights[:, np.newaxis], axis=0)
    def _weighted_average_2d(params_list):
        # Для двумерных параметров (transmat_, emissionprob_)
        params_array = np.array([np.exp(params) for params in params_list])
        return np.sum(params_array * weights[:, np.newaxis, np.newaxis], axis=0)
    
    def _check_probabilities(probs, name):
        sums = np.sum(probs, axis=1)
        if not np.allclose(sums, 1.0, rtol=1e-5):
            raise ValueError(f"Сумма вероятностей в {name} не равна 1: {sums}")

    if not models or not all(isinstance(m, hmm.CategoricalHMM) for m in models):
        raise ValueError("Входной список должен содержать модели CategoricalHMM")
    
    # Проверка и нормализация весов
    if weights is not None:
        if len(weights) != len(models):
            raise ValueError("Количество весов должно соответствовать количеству моделей")
        if not all(w > 0 for w in weights):
            raise ValueError("Все веса должны быть положительными")
        # Нормализуем веса
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(models)) / len(models)
    
    # Проверяем структуру моделей
    n_components = models[0].n_components
    n_features = models[0].n_features
    
    if not all(m.n_components == n_components and m.n_features == n_features for m in models):
        raise ValueError("Все модели должны иметь одинаковое количество состояний и признаков")
            
    # Усредняем
    avg_startprob = _weighted_average_1d([m.startprob_ for m in models])
    avg_startprob = _normalize(avg_startprob)
    
    avg_transmat = _weighted_average_2d([m.transmat_ for m in models])
    avg_transmat = _normalize(avg_transmat, axis=1)
    
    avg_emissionprob = _weighted_average_2d([m.emissionprob_ for m in models])
    avg_emissionprob = _normalize(avg_emissionprob, axis=1)
    
        
    if not np.allclose(np.sum(avg_startprob), 1.0, rtol=1e-5):
        raise ValueError(f"Сумма начальных вероятностей не равна 1: {np.sum(avg_startprob)}")    
    _check_probabilities(avg_transmat, "матрице переходов")
    _check_probabilities(avg_emissionprob, "матрице эмиссий")
    

    avg_model = hmm.CategoricalHMM(n_components=n_components)
    avg_model.startprob_ = avg_startprob
    avg_model.transmat_ = avg_transmat
    avg_model.emissionprob_ = avg_emissionprob

    return avg_model


def fit_model(chunk):
    model = hmm.CategoricalHMM(n_components=2)
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

def test_model(model, labels):
    pass

def main():
    args = utils.parse_args()
    # Data Frame
    bed_df = utils.read_bed(args.bed)
    
    # Mapping from chrom names to np arrays
    fasta = utils.sequences_to_arrays(utils.read_fasta(args.fasta, args.n_chroms))
    
    # Ground truth
    genome_masks = utils.mask_regions(fasta, bed_df)
    
    chrom_models = []
    chrom_lengths = []
    for chrom, sequence in tqdm(fasta.items(), desc="Split-training over chromosomes"):
        chrom_models.append(split_training(sequence, args.cpu))
        chrom_lengths.append(len(sequence))

    aggregated_model = average_hmm_models(chrom_models, chrom_lengths)

    predicted = parallel_predict(aggregated_model, fasta, args.cpu)
    # pr = list(predicted.values())
    # print(pr[0][0:100], pr[0][1000:1100], pr[0][10000:10100], pr[1][0:100], pr[1][1000:1100], pr[1][10000:10100], sep='\n')
    utils.write_bed(predicted, args.out)





if __name__ == '__main__':
    main()