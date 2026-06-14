import numpy as np

class LVQ:
    """
    Model Learning Vector Quantization (LVQ) buatan sendiri menggunakan NumPy.
    Digunakan untuk klasifikasi multi-kelas kondisi fisik mahasiswa:
    0 = Bugar (Energi Tinggi)
    1 = Normal (Energi Sedang)
    2 = Lelah (Energi Rendah)
    """
    def __init__(self, input_size=3, num_classes=3):
        self.input_size = input_size
        self.num_classes = num_classes
        # Inisialisasi prototipe rujukan awal secara terstruktur/deterministik
        self.prototypes = np.array([
            [0.80, 0.80, 0.80],  # Prototipe Kelas 0: Bugar
            [0.60, 0.60, 0.60],  # Prototipe Kelas 1: Normal
            [0.40, 0.40, 0.40]   # Prototipe Kelas 2: Lelah
        ], dtype=float)
        
    def train(self, X, y, epochs=100, lr=0.1):
        # Proses pelatihan kompetitif untuk menggeser prototipe
        history_loss = []
        
        for epoch in range(epochs):
            errors = 0
            # Pengurangan laju pemelajaran (decaying learning rate) secara linier
            current_lr = lr * (1.0 - (epoch / epochs))
            
            # Iterasi setiap sampel data latih
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                
                # 1. Forward Phase: Hitung jarak Euclidean kuadrat ke semua prototipe kelas
                distances = np.sum((self.prototypes - x) ** 2, axis=1)
                
                # 2. Competitive Phase: Pilih prototipe terdekat (Winner-Take-All)
                winner_idx = np.argmin(distances)
                
                # 3. Backward Phase: Pembaruan prototipe pemenang
                if winner_idx == target:
                    # Jika kelas benar (cocok), geser prototipe mendekati input
                    self.prototypes[winner_idx] += current_lr * (x - self.prototypes[winner_idx])
                else:
                    # Jika kelas salah (mismatch), geser prototipe menjauhi input
                    self.prototypes[winner_idx] -= current_lr * (x - self.prototypes[winner_idx])
                    errors += 1
            
            # Catat persentase ketidakcocokan kelas (mismatch rate) pada epoch ini
            mismatch_rate = errors / len(X)
            history_loss.append(mismatch_rate)
            
        return history_loss
        
    def predict(self, x):
        # Jarak Euclidean minimum
        x = np.asarray(x).flatten()
        distances = np.sum((self.prototypes - x) ** 2, axis=1)
        predicted_class = np.argmin(distances)
        return int(predicted_class), distances
