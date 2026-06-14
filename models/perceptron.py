import numpy as np

class Perceptron:
    """
    Model Perceptron untuk klasifikasi biner Safe Day / Risky Day.
    Target:
    0 = Safe Day (Hari Aman)
    1 = Risky Day (Hari Berisiko Tinggi)
    """

    def __init__(self, input_size, learning_rate=0.1, epochs=50):
        # Inisialisasi hyperparameter model
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Inisialisasi bobot awal dengan nol sepanjang ukuran input
        self.weights = np.zeros(input_size)
        # Inisialisasi bias awal dengan 0.0
        self.bias = 0.0
        # List untuk menyimpan riwayat error total setiap epoch
        self.error_history = []

    def activation(self, value):
        # Fungsi aktivasi bipolar step: return 1 jika >= 0, return 0 jika < 0
        return 1 if value >= 0 else 0

    def fit(self, X, y):
        # Proses pelatihan model menggunakan aturan delta Perceptron
        for _ in range(self.epochs):
            total_error = 0

            # Iterasi setiap sampel data latih
            for xi, target in zip(X, y):
                # 1. Forward Phase: Hitung penjumlahan berbobot ditambah bias
                linear_output = np.dot(xi, self.weights) + self.bias
                # Terapkan fungsi aktivasi step
                prediction = self.activation(linear_output)

                # 2. Hitung Error
                error = target - prediction
                total_error += abs(error)

                # 3. Backward Phase: Update bobot dan bias jika ada error
                # Rumus update: w_baru = w_lama + lr * error * xi
                self.weights += self.learning_rate * error * xi
                # Rumus update bias: b_baru = b_lama + lr * error
                self.bias += self.learning_rate * error

            # Catat total error absolut pada epoch ini
            self.error_history.append(total_error)

    def predict(self, X):
        # Melakukan prediksi kelas (0 atau 1) untuk input baru
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def predict_score(self, X):
        # Mengubah output linier menjadi persentase skor risiko (0-100%) menggunakan Sigmoid
        linear_output = np.dot(X, self.weights) + self.bias
        score = 1 / (1 + np.exp(-linear_output))
        return float(score * 100)
