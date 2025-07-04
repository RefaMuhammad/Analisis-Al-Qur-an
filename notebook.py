# %% [markdown]
# # **Tahap 1: Hitung Entropi per Ayat**

# %% [markdown]
# ### **Apa itu Entropi di Sini?**
# - Entropi Shannon mengukur seberapa acak atau bervariasi susunan kata dalam suatu teks.
# - Nilai entropi lebih tinggi → teks lebih bervariasi dan kompleks
# - Nilai entropi lebih rendah → teks lebih repetitif atau terstruktur sederhana

# %% [markdown]
# ## **Import Library**

# %%
import pandas as pd
import re
from collections import Counter
from math import log2
import matplotlib.pyplot as plt


# %% [markdown]
# ## **Memuat Dataset**

# %%
df = pd.read_csv('quran_indonesia.csv')

# %%
# Fungsi preprocessing: hapus tanda baca, lowercase, tokenisasi
def preprocess(text, level='word'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    return text.split() if level == 'word' else list(text.replace(" ", ""))

# %% [markdown]
# ## **Preprocessing Data**

# %%
# Fungsi hitung entropi
def shannon_entropy(tokens):
    total = len(tokens)
    if total == 0:
        return 0
    freqs = Counter(tokens)
    probs = [count / total for count in freqs.values()]
    return -sum(p * log2(p) for p in probs)

# %% [markdown]
# ## **Hitung Entropi**

# %%
# Hitung entropi per ayat (berdasarkan token kata)
df['tokens'] = df['teks'].apply(lambda x: preprocess(x, level='word'))
df['entropy'] = df['tokens'].apply(shannon_entropy)

# %%
# Lihat hasilnya
print(df[['surah', 'ayat', 'teks', 'entropy']])

# %% [markdown]
# ## **A. Hitung Rata-rata Entropi per Surat**

# %%
entropi_per_surat = df.groupby('surah')['entropy'].mean().reset_index()
print(entropi_per_surat)


# %% [markdown]
# ## **B. Visualisasi**

# %%

plt.plot(entropi_per_surat['surah'], entropi_per_surat['entropy'], marker='o')
plt.xlabel("Surah")
plt.ylabel("Entropi Rata-Rata")
plt.title("Entropi Shannon per Surat")
plt.grid()
plt.show()

# %% [markdown]
# ## **Insight**
# 1. Surat Awal (Surah 1–60)
# - Umumnya punya entropi tinggi (sekitar 4.0–4.6)
# - Ini menunjukkan:
#     - Kosakata lebih bervariasi
#     - Struktur ayat lebih naratif atau panjang
#     - Kemungkinan besar: Surat Madaniyah yang memang lebih panjang dan deskriptif
# 2. Surat Akhir (Surah 70 ke atas)
# - Rata-rata entropinya turun ke sekitar 2.5–3.5
# - Ini menunjukkan:
#     - Kosakata lebih sederhana
#     - Banyak pengulangan frasa atau tema
#     - Kemungkinan besar ini adalah Surat Makkiyah yang lebih pendek dan padat
# 
# 
# ## **Hipotesis yang Muncul:**

# %% [markdown]
# ## **Analisis Korelasi Panjang Ayat vs Entropi**

# %%
df['panjang_ayat'] = df['tokens'].apply(len)

plt.scatter(df['panjang_ayat'], df['entropy'], alpha=0.5)
plt.xlabel('Panjang Ayat (jumlah kata)')
plt.ylabel('Entropi')
plt.title('Panjang Ayat vs Entropi')
plt.grid()
plt.show()


# %% [markdown]
# ### **Insight Utama**
# 1. Korelasi Positif antara Panjang dan Entropi
# Semakin panjang sebuah ayat (dalam jumlah kata), semakin tinggi entropinya
#     - Makna: Ayat-ayat panjang cenderung memiliki keragaman kata yang lebih besar
#     - Hal ini logis karena lebih banyak kata → peluang lebih besar untuk variasi kata → entropi meningkat
# 
# 2. Kenaikan Cepat di Awal, Lalu Melandai (Kurikulum Log-Normal)
# Terlihat bentuk seperti kurva logaritmik atau melandai setelah titik tertentu. Artinya:
#     - Pada awalnya, penambahan beberapa kata langsung meningkatkan entropi secara drastis
#     - Tapi setelah melewati sekitar 50–60 kata, penambahan kata tidak banyak meningkatkan entropi lagi
#     - Ini menunjukkan adanya batas alami dalam kompleksitas linguistik per ayat
# 
#     Kemungkinan besar karena ayat panjang mulai mengulang atau menyusun struktur retoris serupa, jadi entropi tidak terus meningkat linear.
# 
# 3. Outlier Menarik
# Ada beberapa titik (ayat) yang:
#     - Panjang tapi entropinya rendah → kemungkinan ada pengulangan kata
#     - Pendek tapi entropinya tinggi → kemungkinan menggunakan kata-kata unik yang sangat variatif


