"""
Этот файл содержит имплементацию класса CpGModel. Ее основные
методы:

    - analyze_sequence: применение алгоритма Витерби с матрицами
                        вероятности, установленными в модели
                        для вычисления оптимального пути (а именно
                        идентификации каждого нуклеотида как
                        принадлежащего CpG островку или нет)

    - baum_welch_training: использование алгоритма Баума-Велша для
                           "обучения" модели и подбора оптимальных
                           вероятностей переходов и эмиссии.

Конструктор from_preproc используется для того, чтобы
инициализировать модель из известных матриц переходов. Например,
в тех случаях, когда Вы уже обучили модель и хотите заново
запустить скрипт с параметрами, полученными после обучения или
Вы хотите попробовать запустить алгоритм Витерби с самостоятельно
установленными вероятностями переходов.

parallel_train_model - метод, позволяющий эффективно запускать
обучение на многоядерной машине
"""

import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CpGModel:
    """
    :var states: Пространство возможных состояний модели
    :var state_to_idx: Маппинг состояний и их индексов
    :var observations: Пространство возможных явлений
    :var observations_map: Маппинг явлений и их индексов
    :var transition_probabilities: Матрица вероятностей переходов
    :var emission_probabilities: Матрица вероятностей эмиссии
    :var log_transition_probs: Логарифм от матрицы переходов. Используется
        для того, чтобы избежать ошибки деления на нулевую вероятность.
    :var log_emission_probs: Логарифм от матрицы состояний. Используется
        для того, чтобы избежать ошибки деления на нулевую вероятность.
    """

    def __init__(self):
        # Initialize states: CpG, Non-CpG
        self.transition_probabilities = np.random.dirichlet(np.ones(2), size=2)  # Random initialization with Dirichlet distribution
        self.emission_probabilities = np.random.dirichlet(np.ones(4), size=2)  # Random initialization with Dirichlet distribution
        self._init()

    def _init(self):
        logging.info("Initializing CpGModel with states and probabilities.")
        self.states = [True, False]
        self.state_to_idx = {True: 0, False: 1}
        self.observations = ['A', 'C', 'G', 'T']
        self.observations_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # Normalize probabilities
        self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True)
        self.emission_probabilities /= self.emission_probabilities.sum(axis=1, keepdims=True)

        # Use log probabilities to avoid numerical underflow
        self.log_transition_probs = np.log(self.transition_probabilities)
        self.log_emission_probs = np.log(self.emission_probabilities)

    @classmethod
    def from_preproc(cls, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """
        Создать экземпляр модели из предварительно посчитанных или вручную заданных матриц

        :param transition_probabilities:
            | Матрица вероятностей переходов между состояниями в формате:
            | [[CG->CG, CG->NonCG],
            | [NonCG->NonCG, NonCG->CG]]

        :param emission_probabilities:
            | Матрица вероятностей эмиссии в формате:
            | ----- : -A- : -C- : -G- : -T- :
            |   CG  :  -  :  -  :  -  :  -  :
            | nonCG :  -  :  -  :  -  :  -  :
        """
        inst = cls()
        inst.transition_probabilities = transition_probabilities
        inst.emission_probabilities = emission_probabilities
        inst._init()
        return inst

    def log_sum_exp(self, log_values):
        max_log = np.max(log_values)
        return max_log + np.log(np.sum(np.exp(log_values - max_log)))

    def baum_welch_training(self, sequence, num_iterations=50):
        n_states = len(self.states)
        seq_len = len(sequence)
        n_observations = len(self.observations)

        seq_converted = [self.observations_map[c] for c in sequence.upper() if c in self.observations_map.keys()]
        epsilon = 1e-6  # Adjusted epsilon for more stability

        for iteration in range(num_iterations):
            logging.info(f"Iteration {iteration + 1}")
            alpha = self._forward_algorithm(sequence)
            beta = self._backward_algorithm(sequence)
            xi = np.full((n_states, n_states, seq_len - 1), -np.inf)
            gamma = np.full((n_states, seq_len), -np.inf)

            # Compute xi and gamma using log-space
            for t in range(seq_len - 1):
                denom = self.log_sum_exp([alpha[i, t] + beta[i, t] for i in range(n_states)])
                for i in range(n_states):
                    gamma[i, t] = alpha[i, t] + beta[i, t] - denom
                    for j in range(n_states):
                        xi[i, j, t] = alpha[i, t] + self.log_transition_probs[i, j] + \
                                      np.log(self.emission_probabilities[j, seq_converted[t + 1]]) + beta[
                                          j, t + 1] - denom

            # Last gamma
            denom = self.log_sum_exp([alpha[i, -1] + beta[i, -1] for i in range(n_states)])
            for i in range(n_states):
                gamma[i, -1] = alpha[i, -1] + beta[i, -1] - denom

            # Re-estimate transition probabilities using log-space smoothing
            for i in range(n_states):
                for j in range(n_states):
                    numer = self.log_sum_exp([xi[i, j, t] for t in range(seq_len - 1)])
                    denom = self.log_sum_exp([gamma[i, t] for t in range(seq_len - 1)])
                    self.transition_probabilities[i, j] = np.exp(numer - denom)

            # Re-estimate emission probabilities using log-space smoothing
            for i in range(n_states):
                for k in range(n_observations):
                    numer = self.log_sum_exp([gamma[i, t] for t in range(seq_len) if seq_converted[t] == k])
                    denom = self.log_sum_exp([gamma[i, t] for t in range(seq_len)])
                    self.emission_probabilities[i, k] = np.exp(numer - denom)

            # Normalize probabilities
            self.transition_probabilities /= self.transition_probabilities.sum(axis=1, keepdims=True) + epsilon
            self.emission_probabilities /= self.emission_probabilities.sum(axis=1, keepdims=True) + epsilon

            # Update log probabilities
            self.log_transition_probs = np.log(self.transition_probabilities + epsilon)
            self.log_emission_probs = np.log(self.emission_probabilities + epsilon)

            logging.info(f"Updated transition probabilities: {self.transition_probabilities}")
            logging.info(f"Updated emission probabilities: {self.emission_probabilities}")

    def analyze_sequence(self, sequence: str):
        """
        Анализировать нуклеотидную последовательность с помощью алгоритма Витерби.

        :param sequence: Нуклеотидная последовательность.
        :return: Массив с предсказанным состоянием для каждого нуклеотида.
        """
        n_states = len(self.states)
        seq_len = len(sequence)

        seq_converted = [self.observations_map[c] for c in sequence.upper()]
        viterbi_matrix = np.zeros((n_states, seq_len))
        backpointer = np.zeros((n_states, seq_len), dtype=int)

        # Initialization using log probabilities
        for state_idx in range(n_states):
            viterbi_matrix[state_idx, 0] = self.log_emission_probs[state_idx, seq_converted[0]]
            backpointer[state_idx, 0] = 0

        # Iteration using log probabilities
        for nuc_idx in range(1, seq_len):
            for state_idx in range(n_states):
                cur_transition_log_probs = [
                    viterbi_matrix[prev_state, nuc_idx - 1] + self.log_transition_probs[prev_state, state_idx]
                    for prev_state in range(n_states)
                ]
                max_transition_log_prob = max(cur_transition_log_probs)
                viterbi_matrix[state_idx, nuc_idx] = max_transition_log_prob + self.log_emission_probs[
                    state_idx, seq_converted[nuc_idx]]
                backpointer[state_idx, nuc_idx] = np.argmax(cur_transition_log_probs)

        # Termination: Finding the best last state
        last_state = np.argmax(viterbi_matrix[:, -1])
        best_path = [last_state]

        # Tracking back to find the best path
        for t in range(seq_len - 1, 0, -1):
            last_state = backpointer[last_state, t]
            best_path.append(last_state)

        # Reverse the path to get the correct order
        best_path = best_path[::-1]

        # Convert state indices back to state names
        best_path_states = [self.states[int(state)] for state in best_path]
        return best_path_states

    def _forward_algorithm(self, sequence):
        n_states = len(self.states)
        seq_len = len(sequence)
        alpha = np.full((n_states, seq_len), -np.inf)  # Initialize with log-space

        seq_converted = [self.observations_map[c] for c in sequence.upper() if c in self.observations_map.keys()]

        # Initialization
        for state in range(n_states):
            alpha[state, 0] = np.log(self.emission_probabilities[state, seq_converted[0]])

        # Induction using log-sum-exp
        for t in range(1, seq_len):
            for j in range(n_states):
                alpha[j, t] = np.log(self.emission_probabilities[j, seq_converted[t]]) + self.log_sum_exp(
                    [alpha[i, t - 1] + self.log_transition_probs[i, j] for i in range(n_states)])

        return alpha

    def _backward_algorithm(self, sequence):
        n_states = len(self.states)
        seq_len = len(sequence)
        beta = np.full((n_states, seq_len), -np.inf)  # Initialize with log-space

        seq_converted = [self.observations_map[c] for c in sequence.upper()]

        # Initialization
        beta[:, -1] = 0  # log(1) = 0

        # Induction using log-sum-exp
        for t in reversed(range(seq_len - 1)):
            for i in range(n_states):
                beta[i, t] = self.log_sum_exp([
                    self.log_transition_probs[i, j] + np.log(self.emission_probabilities[j, seq_converted[t + 1]]) +
                    beta[j, t + 1]
                    for j in range(n_states)
                ])

        return beta


def parallel_train_model(model: CpGModel, sequences: list[str], num_iterations=50, num_workers=4):
    """
    Функция, осуществляющая обучение модели с помощью параллелизации процессов,
    повышающая эффективность использования CPU.

    :param model: Инициализированная CpG модель
    :param sequences: Список нуклеотидных последовательностей, которые будут использованы в обучении
    :param num_iterations: Количество итераций алгоритма Баума-Велша
    :param num_workers: Количество одновременных процессов
    :return:
    """

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(train_on_segment, copy.deepcopy(model), segment, num_iterations)
            for _, segment in sequences
        ]
        models = []
        for future in as_completed(futures):
            models.append(future.result())

    # Aggregate the results by averaging the parameters from all models
    aggregated_transition_probabilities = np.mean([m.transition_probabilities for m in models], axis=0)
    aggregated_emission_probabilities = np.mean([m.emission_probabilities for m in models], axis=0)

    model.transition_probabilities = aggregated_transition_probabilities
    model.emission_probabilities = aggregated_emission_probabilities
    model.log_transition_probs = np.log(model.transition_probabilities)
    model.log_emission_probs = np.log(model.emission_probabilities)

    logging.info("Aggregated transition probabilities:")
    logging.info(model.transition_probabilities)
    logging.info("Aggregated emission probabilities:")
    logging.info(model.emission_probabilities)


def train_on_segment(model: CpGModel, segment, num_iterations):
    model.baum_welch_training(segment, num_iterations)
    return model
