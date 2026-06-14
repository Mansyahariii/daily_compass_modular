import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Impor model buatan sendiri dan dataset latihan
from models.perceptron import Perceptron
from models.lvq import LVQ
from data import get_perceptron_dataset, get_lvq_dataset
from utils import normalize_risk_inputs, normalize_lvq_inputs, get_daily_mode

@st.cache_resource
def train_models(perceptron_lr, perceptron_epochs, lvq_lr, lvq_epochs):
    # Melakukan pelatihan Perceptron dan LVQ berdasarkan parameter dari UI sidebar
    X_p, y_p = get_perceptron_dataset()
    perceptron = Perceptron(
        input_size=4,
        learning_rate=perceptron_lr,
        epochs=perceptron_epochs
    )
    perceptron.fit(X_p, y_p)

    X_l, y_l = get_lvq_dataset()
    lvq = LVQ(
        input_size=3,
        num_classes=3
    )
    history_loss = lvq.train(X_l, y_l, epochs=lvq_epochs, lr=lvq_lr)

    return perceptron, lvq, history_loss

def draw_neural_network(layer_sizes, node_labels=None, title="Arsitektur Jaringan Syaraf Tiruan"):
    # Fungsi pembantu untuk merender arsitektur jaringan saraf Perceptron dan LVQ
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis('off')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Koordinat X untuk setiap layer jaringan
    x_coords = np.linspace(0.15, 0.85, len(layer_sizes))
    
    # Render node per layer
    node_positions = []
    for i, size in enumerate(layer_sizes):
        y_coords = np.linspace(0.15, 0.85, size) if size > 1 else np.array([0.5])
        layer_pos = [(x_coords[i], y) for y in y_coords]
        node_positions.append(layer_pos)
        
        # Pewarnaan node
        if i == 0:
            color = '#38bdf8'  # Biru untuk Input
            label_prefix = 'Input'
        elif i == len(layer_sizes) - 1:
            color = '#34d399'  # Hijau untuk Output
            label_prefix = 'Output'
        else:
            color = '#a78bfa'  # Ungu untuk Competitive (LVQ)
            label_prefix = 'Competitive'
            
        for j, (x, y) in enumerate(layer_pos):
            circle = plt.Circle((x, y), 0.05, color=color, zorder=3, ec='#1e293b', lw=1.5)
            ax.add_patch(circle)
            
            node_name = node_labels[i][j] if (node_labels and i < len(node_labels) and node_labels[i] is not None and j < len(node_labels[i])) else f"{label_prefix[0]}{j+1}"
            ax.text(x, y, node_name, ha='center', va='center', color='#0f172a', fontweight='bold', fontsize=7, zorder=4)
            
            if j == size - 1:
                ax.text(x, 0.95, f"{label_prefix}", ha='center', va='bottom', color='#94a3b8', fontweight='bold', fontsize=9)

    # Menggambar garis koneksi antar node
    for i in range(len(layer_sizes) - 1):
        for x1, y1 in node_positions[i]:
            for x2, y2 in node_positions[i+1]:
                ax.annotate("", xy=(x2 - 0.05, y2), xytext=(x1 + 0.05, y1),
                            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2, shrinkA=0, shrinkB=0))
                
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=12)
    return fig

# Pengaturan halaman web Streamlit
st.set_page_config(
    page_title="Daily Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling CSS bertema premium dark glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
    }
    
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15) !important;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Judul Utama Dashboard dengan SVG Icon Compass
st.markdown("""
<h1 class='main-title' style='display: flex; align-items: center; justify-content: center;'>
    <svg xmlns="http://www.w3.org/2000/svg" width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="url(#gradient-accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 14px;">
        <defs>
            <linearGradient id="gradient-accent" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#38bdf8" />
                <stop offset="100%" stop-color="#a78bfa" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="10"></circle>
        <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="url(#gradient-accent)" fill-opacity="0.25"></polygon>
    </svg>
    Daily Compass
</h1>
""", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistem Rekomendasi Keputusan Harian Mahasiswa Berdasarkan Energi, Budget, dan Beban Aktivitas Menggunakan Perceptron & Learning Vector Quantization (LVQ)</p>", unsafe_allow_html=True)

st.divider()

# Sidebar input data mentah pengguna
st.sidebar.header("Input Kondisi Harian")
budget = st.sidebar.slider("Budget hari ini (Rp)", min_value=0, max_value=50000, value=20000, step=1000)
hunger = st.sidebar.slider("Tingkat lapar (0-10)", min_value=0, max_value=10, value=5)
deadline = st.sidebar.slider("Deadline terdekat (hari lagi)", min_value=0, max_value=7, value=2)
tasks = st.sidebar.slider("Jumlah tugas hari ini", min_value=0, max_value=10, value=4)
sleep = st.sidebar.slider("Jam tidur semalam (jam)", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
mood = st.sidebar.slider("Mood pagi (0-10)", min_value=0, max_value=10, value=6)
caffeine = st.sidebar.slider("Konsumsi kafein (gelas)", min_value=0, max_value=5, value=1)

# Sidebar input hiperparameter pelatihan
st.sidebar.divider()
st.sidebar.header("Parameter Training")
perceptron_lr = st.sidebar.selectbox("Learning Rate Perceptron", [0.01, 0.05, 0.1, 0.2], index=2)
perceptron_epochs = st.sidebar.selectbox("Epoch Perceptron", [10, 30, 50, 100], index=2)
lvq_lr = st.sidebar.selectbox("Learning Rate LVQ", [0.01, 0.05, 0.1, 0.2], index=2)
lvq_epochs = st.sidebar.selectbox("Epoch LVQ", [50, 100, 200, 500], index=1)
st.sidebar.markdown("<br><p style='text-align: center; color: #38bdf8; font-weight: bold; font-size: 0.95rem; border: 1px solid rgba(56, 189, 248, 0.3); padding: 8px; border-radius: 8px; background: rgba(56, 189, 248, 0.05);'>🧭 Analisis Dihitung Real-time</p>", unsafe_allow_html=True)

# Latih model secara dinamis
with st.spinner("Melatih model JST..."):
    perceptron_model, lvq_model, lvq_loss_history = train_models(
        perceptron_lr, perceptron_epochs, lvq_lr, lvq_epochs
    )

# Tata letak Tab
tab1, tab2, tab3, tab4 = st.tabs(["Hasil Rekomendasi", "Grafik Training", "Dataset", "Detail Model"])

with tab1:
    st.subheader("Hasil Analisis Harian")
    # Perhitungan analisis JST dijalankan secara otomatis (Real-time)
    if True:
        # Preprocessing & Normalisasi Input
        risk_input = normalize_risk_inputs(budget, hunger, deadline, tasks)
        lvq_input = normalize_lvq_inputs(sleep, mood, caffeine)

        # Proses klasifikasi Perceptron
        risk_prediction = perceptron_model.predict(risk_input)
        risk_score = perceptron_model.predict_score(risk_input)

        # Proses klasifikasi kompetitif LVQ
        lvq_prediction, distances = lvq_model.predict(lvq_input)
        
        energy_labels = ["Bugar (High) ⚡", "Normal (Moderate) ⚖️", "Lelah (Low) 💤"]
        energy_status = energy_labels[lvq_prediction]
        energy_color = "#34d399" if lvq_prediction == 0 else ("#f59e0b" if lvq_prediction == 1 else "#ef4444")

        # Tentukan Mode Rekomendasi
        mode, recommendations = get_daily_mode(lvq_prediction, risk_prediction)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>⚡ Physical State (LVQ)</div>
                <div class='metric-value' style='color: {energy_color}; font-size: 1.8rem;'>{energy_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            risk_label = "Risky Day 🚨" if risk_prediction == 1 else "Safe Day 🛡️"
            risk_color = "#ff0844" if risk_prediction == 1 else "#34d399"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>📅 Daily Risk</div>
                <div class='metric-value' style='color: {risk_color}; font-size: 1.8rem;'>{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>📊 Risk Score</div>
                <div class='metric-value' style='font-size: 1.8rem;'>{risk_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Mode Hari Ini: {mode}")

        # Tampilkan Alert box yang sesuai
        if mode == "Critical Mode":
            st.error("🚨 Kondisi hari ini cukup berat. Batasi pengeluaran dan prioritas utama adalah kesehatan.")
        elif mode == "Alert Mode":
            st.warning("⚠️ Risiko tergolong tinggi tetapi energi memadai. Kerjakan tugas krusial secara bertahap.")
        elif mode == "Focus Mode":
            st.success("🟢 Hari produktif! Energi optimal dan risiko minimal. Sangat cocok menyelesaikan tugas berat.")
        elif mode == "Recovery Mode":
            st.info("🔵 Risiko rendah tetapi tubuh lelah. Ambil waktu istirahat yang cukup sebelum lanjut belajar.")
        else:
            st.info("🟡 Kondisi harian Anda tergolong stabil dan seimbang. Jalankan aktivitas secara normal.")

        st.subheader("Rekomendasi Aktivitas")
        for item in recommendations:
            st.markdown(f"- {item}")

        st.divider()
        st.subheader("🔍 Penjelasan Langkah & Perhitungan Matematika JST")
        with st.expander("Klik untuk melihat detail langkah demi langkah perhitungan rumus JST (Explainable AI)", expanded=False):
            st.markdown("### 📋 Ringkasan Parameter Input & Hasil Normalisasi")
            st.markdown("Tabel di bawah merangkum data input mentah Anda dibandingkan dengan nilai hasil normalisasi ($x_i$) yang diumpankan ke model JST:")
            
            input_summary_data = {
                "Parameter Input": [
                    "Budget Harian (Uang Saku)",
                    "Tingkat Rasa Lapar",
                    "Sisa Waktu Deadline",
                    "Jumlah Beban Tugas",
                    "Waktu Tidur Semalam",
                    "Kondisi Mood Pagi",
                    "Konsumsi Kafein"
                ],
                "Nilai Input Mentah": [
                    f"Rp {budget:,.0f}",
                    f"{hunger} / 10",
                    f"{deadline} hari",
                    f"{tasks} tugas",
                    f"{sleep} jam",
                    f"{mood} / 10",
                    f"{caffeine} gelas"
                ],
                "Nilai Ternormalisasi (x_i)": [
                    f"{risk_input[0]:.3f} (x₁)",
                    f"{risk_input[1]:.3f} (x₂)",
                    f"{risk_input[2]:.3f} (x₃)",
                    f"{risk_input[3]:.3f} (x₄)",
                    f"{lvq_input[0]:.3f} (x₅)",
                    f"{lvq_input[1]:.3f} (x₆)",
                    f"{lvq_input[2]:.3f} (x₇)"
                ],
                "Model Penerima": [
                    "Perceptron (Risk Classifier)",
                    "Perceptron (Risk Classifier)",
                    "Perceptron (Risk Classifier)",
                    "Perceptron (Risk Classifier)",
                    "LVQ (Energy Classifier)",
                    "LVQ (Energy Classifier)",
                    "LVQ (Energy Classifier)"
                ]
            }
            df_input = pd.DataFrame(input_summary_data)
            st.table(df_input)
            
            st.divider()

            # Step 1: Normalisasi
            st.markdown("### 📊 Langkah 1: Normalisasi Data Input")
            st.markdown("Data mentah pengguna dinormalisasi ke rentang $[0.0, 1.0]$ agar model JST dapat memprosesnya dengan adil:")
            latex_str1 = r"""
\begin{aligned}
x_{1} \text{ (Budget Risk)} &= 1 - \frac{\text{budget}}{50000} = 1 - \frac{""" + str(budget) + r"""}{50000} = """ + f"{risk_input[0]:.3f}" + r""" \\
x_{2} \text{ (Hunger Risk)} &= \frac{\text{hunger}}{10} = \frac{""" + str(hunger) + r"""}{10} = """ + f"{risk_input[1]:.3f}" + r""" \\
x_{3} \text{ (Deadline Risk)} &= 1 - \frac{\text{deadline}}{7} = 1 - \frac{""" + str(deadline) + r"""}{7} = """ + f"{risk_input[2]:.3f}" + r""" \\
x_{4} \text{ (Task Load)} &= \frac{\text{tasks}}{10} = \frac{""" + str(tasks) + r"""}{10} = """ + f"{risk_input[3]:.3f}" + r"""
\end{aligned}
"""
            st.latex(latex_str1)

            latex_str2 = r"""
\begin{aligned}
x_{5} \text{ (Sleep Score)} &= \frac{\text{sleep}}{10} = \frac{""" + str(sleep) + r"""}{10} = """ + f"{lvq_input[0]:.3f}" + r""" \\
x_{6} \text{ (Mood Score)} &= \frac{\text{mood}}{10} = \frac{""" + str(mood) + r"""}{10} = """ + f"{lvq_input[1]:.3f}" + r""" \\
x_{7} \text{ (Caffeine Score)} &= 1 - \frac{|\text{caffeine} - 2|}{5} = 1 - \frac{|""" + str(caffeine) + r""" - 2|}{5} = """ + f"{lvq_input[2]:.3f}" + r"""
\end{aligned}
"""
            st.latex(latex_str2)

            # Step 2: Perceptron
            st.markdown("### 🧭 Langkah 2: Perhitungan Model Perceptron (Risk Classifier)")
            st.markdown("Perceptron mengalikan input risiko dengan bobot ($w$) masing-masing ditambah nilai bias ($b$):")
            st.latex(r"v = \sum_{i=1}^4 (x_i \cdot w_i) + b = (x_1 \cdot w_1) + (x_2 \cdot w_2) + (x_3 \cdot w_3) + (x_4 \cdot w_4) + b")
            
            w_p = perceptron_model.weights
            b_p = perceptron_model.bias
            v_p = np.dot(risk_input, w_p) + b_p
            
            latex_str3 = r"""
\begin{aligned}
v &= (""" + f"{risk_input[0]:.3f} \cdot {w_p[0]:.4f}" + r""") + (""" + f"{risk_input[1]:.3f} \cdot {w_p[1]:.4f}" + r""") + (""" + f"{risk_input[2]:.3f} \cdot {w_p[2]:.4f}" + r""") + (""" + f"{risk_input[3]:.3f} \cdot {w_p[3]:.4f}" + r""") + (""" + f"{b_p:.4f}" + r""") \\
v &= """ + f"{risk_input[0]*w_p[0]:.4f} + {risk_input[1]*w_p[1]:.4f} + {risk_input[2]*w_p[2]:.4f} + {risk_input[3]*w_p[3]:.4f} + {b_p:.4f}" + r""" \\
v &= """ + f"{v_p:.4f}" + r"""
\end{aligned}
"""
            st.latex(latex_str3)
            
            st.markdown(r"""
            **Fungsi Aktivasi Step Bipolar:**
            $$y = \begin{cases} 1 & \text{jika } v \ge 0 \\ 0 & \text{jika } v < 0 \end{cases}$$
            Karena nilai $v = """ + f"{v_p:.4f}" + r"""$ (maka $v \ge 0$ bernilai **""" + str(v_p >= 0) + r"""**), output Perceptron adalah: **""" + str(risk_prediction) + r""" (""" + ("Risky Day 🚨" if risk_prediction == 1 else "Safe Day 🛡️") + r""")**.
            """)

            # Step 3: LVQ
            st.markdown("### 🎯 Langkah 3: Perhitungan Model LVQ (Energy Classifier)")
            st.markdown("LVQ menghitung Jarak Euclidean dari input fisik ke 3 bobot prototipe kelas ($W_0$: Bugar, $W_1$: Normal, $W_2$: Lelah):")
            st.latex(r"d_j = \sqrt{\sum_{i=1}^3 (x_{i+4} - w_{ji})^2}")
            
            d0 = np.sqrt(distances[0])
            d1 = np.sqrt(distances[1])
            d2 = np.sqrt(distances[2])
            
            latex_str4 = r"""
\begin{aligned}
d_0 \text{ (Bugar)} &= \sqrt{(""" + f"{lvq_input[0]:.3f} - {lvq_model.prototypes[0][0]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[1]:.3f} - {lvq_model.prototypes[0][1]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[2]:.3f} - {lvq_model.prototypes[0][2]:.3f}" + r"""\big)^2} = """ + f"{d0:.4f}" + r""" \\
d_1 \text{ (Normal)} &= \sqrt{(""" + f"{lvq_input[0]:.3f} - {lvq_model.prototypes[1][0]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[1]:.3f} - {lvq_model.prototypes[1][1]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[2]:.3f} - {lvq_model.prototypes[1][2]:.3f}" + r"""\big)^2} = """ + f"{d1:.4f}" + r""" \\
d_2 \text{ (Lelah)} &= \sqrt{(""" + f"{lvq_input[0]:.3f} - {lvq_model.prototypes[2][0]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[1]:.3f} - {lvq_model.prototypes[2][1]:.3f}" + r"""\big)^2 + \big(""" + f"{lvq_input[2]:.3f} - {lvq_model.prototypes[2][2]:.3f}" + r"""\big)^2} = """ + f"{d2:.4f}" + r"""
\end{aligned}
"""
            st.latex(latex_str4)
            
            winner_idx = np.argmin([d0, d1, d2])
            winner_text = ["Bugar (High Energy) ⚡", "Normal (Moderate Energy) ⚖️", "Lelah (Low Energy) 💤"][winner_idx]
            
            st.markdown(rf"""
            Jarak terkecil adalah $d_{winner_idx} = {min(d0, d1, d2):.4f}$, sehingga prototipe pemenangnya adalah **Prototipe {winner_idx} ({winner_text})**.
            """)

            # Step 4: Decision Integration
            st.markdown("### 🧩 Langkah 4: Logika Penggabungan Keputusan Final")
            st.markdown(rf"""
            Sistem mengambil output dari kedua model JST untuk menentukan mode aktivitas harian:
            *   **Output Perceptron (Risiko)**: `{'Risky Day 🚨' if risk_prediction == 1 else 'Safe Day 🛡️'}`
            *   **Output LVQ (Stamina)**: `{winner_text}`
            
            Berdasarkan tabel aturan gabungan keputusan:
            """)
            
            st.markdown("""
            | Status Risiko (Perceptron) | Status Fisik (LVQ) | Mode Keputusan Final |
            | :--- | :--- | :--- |
            | Risky Day | Lelah | **Critical Mode** 🚨 |
            | Risky Day | Bugar / Normal | **Alert Mode** ⚠️ |
            | Safe Day | Bugar | **Focus Mode** 🟢 |
            | Safe Day | Lelah | **Recovery Mode** 🔵 |
            | Safe Day | Normal | **Balanced Mode** 🟡 |
            """)
            
            st.info(f"Hasil kombinasi input Anda menghasilkan keputusan: **{mode}**")
    # Analisis diperbarui secara otomatis setiap kali parameter input pada sidebar diubah
    st.caption("ℹ️ Analisis dan seluruh visualisasi langkah perhitungan matematika di atas diperbarui secara real-time pada setiap perubahan input di sidebar.")

with tab2:
    st.subheader("Visualisasi Kurva Pelatihan Model JST")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        fig1.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')
        ax1.plot(perceptron_model.error_history, color='#38bdf8', linewidth=2.5, label='Total Error')
        ax1.set_title("Kurva Penurunan Error Perceptron", color='white', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Epoch", color='#94a3b8', fontsize=10)
        ax1.set_ylabel("Total Absolut Error", color='#94a3b8', fontsize=10)
        ax1.tick_params(colors='#94a3b8', labelsize=9)
        ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax1.legend(facecolor='#0f172a', edgecolor='none', labelcolor='white')
        st.pyplot(fig1)
        plt.close(fig1)

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#0f172a')
        ax2.plot(lvq_loss_history, color='#a78bfa', linewidth=2.5, label='Mismatch Rate')
        ax2.set_title("Kurva Penurunan Mismatch Rate LVQ", color='white', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Epoch", color='#94a3b8', fontsize=10)
        ax2.set_ylabel("Mismatch Rate (0.0 - 1.0)", color='#94a3b8', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=9)
        ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax2.legend(facecolor='#0f172a', edgecolor='none', labelcolor='white')
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown(f"""
    > **📊 Hasil Evaluasi Akhir:**
    > *   **Perceptron**: Berhasil memisahkan pola biner linier *Safe Day* dan *Risky Day* secara tuntas dalam {perceptron_epochs} epoch.
    > *   **LVQ (Learning Vector Quantization)**: Mengalami konvergensi dengan melatih prototipe rujukan secara kompetitif. Final Mismatch Rate: **{lvq_loss_history[-1]:.4f}** pada {lvq_epochs} epoch.
    """)

with tab3:
    st.subheader("Data Latih Model (Dataset JST)")
    X_p, y_p = get_perceptron_dataset()
    df_p = pd.DataFrame(X_p, columns=["budget_risk", "hunger_risk", "deadline_risk", "task_load"])
    df_p["label"] = y_p
    df_p["keterangan"] = df_p["label"].map({0: "Safe Day", 1: "Risky Day"})
    st.markdown("#### 1. Dataset Perceptron (Bipolar Risk)")
    st.dataframe(df_p, use_container_width=True)

    X_l, y_l = get_lvq_dataset()
    df_l = pd.DataFrame(X_l, columns=["sleep_score", "mood_score", "caffeine_score"])
    df_l["class_target"] = y_l
    df_l["keterangan"] = df_l["class_target"].map({0: "Bugar (High Energy)", 1: "Normal (Moderate Energy)", 2: "Lelah (Low Energy)"})
    st.markdown("#### 2. Dataset LVQ (Physical State Classifier)")
    st.dataframe(df_l, use_container_width=True)

with tab4:
    st.subheader("Matematika & Arsitektur Model JST")
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("### 1. Model Perceptron (Risk Classifier)")
        fig_p = draw_neural_network(
            layer_sizes=[4, 1],
            node_labels=[['Budget\n(x1)', 'Lapar\n(x2)', 'Deadline\n(x3)', 'Tugas\n(x4)'], ['Output\n(y)']],
            title="Diagram Arsitektur Perceptron Bipolar"
        )
        st.pyplot(fig_p)
        plt.close(fig_p)
        st.markdown("""
        Perceptron mengklasifikasikan data masukan dengan fungsi aktivasi bipolar step:
        $$y = \\begin{cases} 1 & \\text{jika } w^T x + b \\ge 0 \\\\ 0 & \\text{jika } w^T x + b < 0 \\end{cases}$$
        """)
        st.write("Weights ($w$):", perceptron_model.weights)
        st.write("Bias ($b$):", perceptron_model.bias)

    with col_arch2:
        st.markdown("### 2. Model LVQ (Competitive Classifier)")
        fig_lvq = draw_neural_network(
            layer_sizes=[3, 3, 1],
            node_labels=[['Tidur\n(x1)', 'Mood\n(x2)', 'Kafein\n(x3)'], ['W0\n(Bugar)', 'W1\n(Normal)', 'W2\n(Lelah)'], ['Class\n(y)']],
            title="Diagram Arsitektur LVQ"
        )
        st.pyplot(fig_lvq)
        plt.close(fig_lvq)
        st.markdown("""
        LVQ mencari prototipe kelas terdekat menggunakan Jarak Euclidean:
        $$d_j = \\sqrt{\\sum_{i=1}^3 (x_i - w_{ji})^2}$$
        """)
        st.write("Prototipe Kelas 0 (Bugar):", lvq_model.prototypes[0])
        st.write("Prototipe Kelas 1 (Normal):", lvq_model.prototypes[1])
        st.write("Prototipe Kelas 2 (Lelah):", lvq_model.prototypes[2])
