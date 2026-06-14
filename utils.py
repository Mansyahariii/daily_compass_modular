import numpy as np

def normalize_risk_inputs(budget, hunger, deadline, tasks):
    # Proses normalisasi data input mentah pengguna ke dalam rentang 0.0 - 1.0 (Perceptron)
    max_budget = 50000
    max_deadline = 7
    max_tasks = 10

    # Budget risk: semakin tipis uang saku, skor risiko finansial mendekati 1.0
    budget_risk = 1 - (min(budget, max_budget) / max_budget)
    # Hunger risk: tingkat lapar (0-10) dibagi 10
    hunger_risk = hunger / 10
    # Deadline risk: semakin mepet tenggat waktu, risiko mendekati 1.0
    deadline_risk = 1 - (min(deadline, max_deadline) / max_deadline)
    # Task load: jumlah tugas kuliah hari ini dibagi 10
    task_load = min(tasks, max_tasks) / max_tasks

    return np.array([budget_risk, hunger_risk, deadline_risk, task_load])

def normalize_lvq_inputs(sleep, mood, caffeine):
    # Proses normalisasi data input mentah pengguna ke dalam rentang 0.0 - 1.0 (LVQ)
    sleep_score = min(sleep, 10) / 10
    mood_score = mood / 10

    # Caffeine score: dibuat fungsi parabola terbalik dengan titik puncak ideal di 2 gelas kopi
    caffeine_score = 1 - abs(caffeine - 2) / 5
    caffeine_score = np.clip(caffeine_score, 0, 1)

    return np.array([sleep_score, mood_score, caffeine_score])

def get_daily_mode(lvq_prediction, risk_prediction):
    """
    Menentukan mode rekomendasi harian berdasarkan kombinasi keluaran LVQ dan Perceptron.
    lvq_prediction: 0 = Bugar, 1 = Normal, 2 = Lelah
    risk_prediction: 0 = Safe Day, 1 = Risky Day
    """
    # 1. Critical Mode: Tubuh lelah dan stresor risiko luar sangat tinggi
    if risk_prediction == 1 and lvq_prediction == 2:
        return "Critical Mode", [
            "Kondisi kamu cukup berat hari ini (Energi Rendah & Risiko Tinggi).",
            "Prioritaskan kebutuhan dasar seperti makan dan istirahat singkat.",
            "Kerjakan satu tugas yang deadline-nya paling dekat selama 25 menit.",
            "Hindari multitasking karena energi sedang rendah."
        ]

    # 2. Alert Mode: Risiko tinggi tetapi stamina fisik masih prima
    if risk_prediction == 1 and lvq_prediction < 2:
        return "Alert Mode", [
            "Hari ini punya risiko cukup tinggi, tapi energi kamu masih memadai.",
            "Kerjakan tugas paling penting lebih dulu.",
            "Gunakan budget hanya untuk kebutuhan utama.",
            "Ambil jeda pendek setiap 25-30 menit."
        ]

    # 3. Focus Mode: Stamina prima dan tidak ada stresor risiko luar yang berarti
    if risk_prediction == 0 and lvq_prediction == 0:
        return "Focus Mode", [
            "Energi kamu sangat bagus dan risiko harian rendah.",
            "Sangat cocok untuk mengerjakan tugas berat yang butuh konsentrasi tinggi.",
            "Gunakan waktu produktif ini semaksimal mungkin.",
            "Tetap kontrol pengeluaran agar kondisi finansial tetap aman."
        ]

    # 4. Recovery Mode: Risiko luar rendah namun stamina fisik drop
    if risk_prediction == 0 and lvq_prediction == 2:
        return "Recovery Mode", [
            "Risiko harian rendah, tetapi tingkat energi Anda sedang drop.",
            "Mulai dari tugas ringan terlebih dahulu untuk memicu motivasi.",
            "Istirahat 15-20 menit sebelum mengerjakan tugas utama.",
            "Hindari tidur larut malam hari ini untuk memulihkan stamina."
        ]

    # 5. Balanced Mode: Kondisi netral
    return "Balanced Mode", [
        "Kondisi harian Anda tergolong cukup stabil.",
        "Kerjakan tugas kuliah dengan tingkat kesulitan sedang.",
        "Jaga ritme seimbang antara belajar, makan, dan istirahat.",
        "Tidak perlu memaksakan semua tugas selesai sekaligus."
    ]
