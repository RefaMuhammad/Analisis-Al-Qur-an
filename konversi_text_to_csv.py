import pandas as pd

# Baca file txt
with open('id.indonesian.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parsing baris
data = []
for line in lines:
    parts = line.strip().split('|')
    if len(parts) == 3:
        surah, ayat, teks = parts
        data.append({
            'surah': int(surah),
            'ayat': int(ayat),
            'teks': teks
        })

# Buat DataFrame
df = pd.DataFrame(data)

# Simpan sebagai CSV
df.to_csv('quran_indonesia.csv', index=False)
print("✅ Berhasil disimpan ke 'quran_indonesia.csv'")
