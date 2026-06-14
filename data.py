import numpy as np


def get_perceptron_dataset():
    """
    Fitur:
    1. budget_risk   : semakin rendah budget, semakin tinggi risiko
    2. hunger_risk   : semakin lapar, semakin tinggi risiko
    3. deadline_risk : semakin dekat deadline, semakin tinggi risiko
    4. task_load     : semakin banyak tugas, semakin tinggi risiko

    Label:
    0 = Safe Day
    1 = Risky Day
    """

    X = np.array([
        [0.90, 0.90, 0.90, 0.90],
        [0.80, 0.70, 0.90, 0.80],
        [0.70, 0.80, 0.80, 0.70],
        [0.60, 0.90, 0.70, 0.80],
        [0.90, 0.60, 0.80, 0.70],

        [0.10, 0.20, 0.10, 0.20],
        [0.20, 0.30, 0.20, 0.30],
        [0.30, 0.20, 0.30, 0.20],
        [0.20, 0.40, 0.20, 0.30],
        [0.40, 0.20, 0.30, 0.20],

        [0.80, 0.80, 0.20, 0.30],
        [0.30, 0.40, 0.90, 0.90],
        [0.60, 0.50, 0.60, 0.60],
        [0.20, 0.80, 0.30, 0.40],
        [0.70, 0.70, 0.70, 0.70],
    ])

    y = np.array([
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        0, 1, 1, 0, 1
    ])

    return X, y


def get_lvq_dataset():
    """
    Fitur:
    1. sleep_score    : jam tidur / 10
    2. mood_score     : mood pagi / 10
    3. caffeine_score : ideal kafein (1 - |caffeine - 2| / 5)

    Label Kelas:
    0 = Bugar (High Energy)
    1 = Normal (Moderate Energy)
    2 = Lelah (Low Energy)
    """

    X = np.array([
        # Kelas 0: Bugar (Tidur cukup, Mood baik, Kafein pas)
        [0.85, 0.90, 0.80],
        [0.80, 0.80, 0.70],
        [0.90, 0.70, 0.80],
        [0.75, 0.85, 0.90],

        # Kelas 1: Normal (Tidur cukup/sedang, Mood sedang, Kafein sedang)
        [0.60, 0.60, 0.60],
        [0.70, 0.50, 0.50],
        [0.55, 0.70, 0.40],
        [0.65, 0.60, 0.50],

        # Kelas 2: Lelah (Tidur kurang, Mood jelek, Kafein kurang/berlebih)
        [0.40, 0.30, 0.20],
        [0.30, 0.40, 0.30],
        [0.20, 0.20, 0.10],
        [0.35, 0.30, 0.30],
    ])

    y = np.array([
        0, 0, 0, 0,  # Bugar
        1, 1, 1, 1,  # Normal
        2, 2, 2, 2   # Lelah
    ])

    return X, y
