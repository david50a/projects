import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
def frequency(text, max_shift=100):
    matches = {}
    for move in range(1, min(max_shift, len(text))):
        count = 0
        for i in range(len(text) - move):
            if text[i] == text[i + move]:
                count += 1
        matches[move] = count
    return matches

ciphertext_raw = """UEKQG VFVMF ODMFT KIMIM YRLQM XCQFF RVOUD RNPAI GJTAH KUJKQ BVZKA
TVETA RFWWQ JRBTQ XSCFY UJBAR GCTNK NVZSD GELYA ZYMDM TUBTQ XVEME
TFBTU TXBTM ZJPQI ULTPZ UKPMH KXQHQ TKWFT KTPUX JFVOQ YYMSM BVPQD
GCQFF RVZUP OEOTA UUWRD KUDQX BVBIT OTPEG OKMPT KIAAI KCTFT GKATQ
CFCXP TVDQD CVIDM TPBTU TXMXE KJWET KNIEM RNIKE IRTXQ JCQFF RVZQP XZLUZ
MYWAP UEMPM EYMDY UKPQD YRQPF UYMDO UDMXU ZKTQD KUZUP OEOTA
UUPQD KZAMB OVKQA LTIWQ GELMN UKBXQ UWEUZ KKIWQ ZYMYF UPWGD MIIZP
SFBTQ XJPQU YZTXM TUEQM QRVPF NVGIU RCLAT KIOAA JJMFA AKJQR UIMUF
MVBET UKIZP CYMZK ULIDQ MFQZS CRTWZ OTMXK GELCG OVBXK GELPA TFBDG
TFNRF NVXMF NFZKA ADIKR GCTMZ JSZQM QKPQN UKBXQ GELFT KEGAG XXZMZ
JDWFT KIEUX RXMFZ UKPUZ MRVPI NVVKA AXWUZ ZFPQD XFWYP UEBRA XXMFF
UJIKS UFLYA XEQZS GELPA TKXQQ VZVFA KMMDK IFZZQ XSMRA XVGAG JFQFU CZTXF
GBMSD KRBOM XVAMU JCQFF RVZQP XZLUZ MYWAP ZFPQD SFBTQ XRVPS GMMTQ
XYIZP UEQFF NVODM TUUAF NVZXU BVLAG ZZVFT KNWAP NRTRM RVISG KWZAY
ZYMHU RCISQ GELVG YKIEX OKBXQ XVLDU JZVST UFLQZ ZVZQP ZYMIA UUIIA RWUQF
NVZDQ JIQPU TXPAA JUQPZ UKSZA CNPMF GNQOW KUKDQ GKCDQ NVEME GELIM
YEWFM ZRTXM LIIUP UWPUY MFWPP GPTUF ZCMDQ JIQPU TXPAA JJIUP NVBTM
TBGAG QZVPX ENWXR CYQFT KIIIM EJWQM XCGXU ZKTQD KUZUP OEOTA UUBAY
EXZMZ JDWFT KIAIT GKPMH KPWGS UKQZK ULZMB XFVOM QVIZP CZVQK KJBQD
JRGIM YSIWU TXLMK YFXAA XJQOW MIIZP SFBTQ XZAFA NRDQE UDMFT OEOSA
UUBAY GBMTQ XJBDA TXMDI NVZQP UVAKA AIODM TUUAF NVZXU BVTUF ZCMDQ
JIQPU TXPAA JROAA JHCMD ZVZAR GCMMS AVNMD ZYMDA TZVFT KNWAP NVZTA
AJMEF GELEG TUMDF NVBTD KVTMD MVWMW ZIMQE ZYMZG ZKZQQ YRZQV AJBNQ
RFEKA AJCDQ RPUGE ZBVAI OKZQB RZMPX OKBXQ XVLDU JZVST UFL"""

clean_text = ''.join(ciphertext_raw.split())
max_shift = 50  # check key lengths up to 20
matches = frequency(clean_text, max_shift=max_shift)

x = np.array(list(matches.keys()))
y = np.array(list(matches.values()))

plt.figure(figsize=(12, 6))
bars = plt.bar(x, y, color='skyblue', edgecolor='k')

# highlight top 5 peaks
top_n = 5
top_indices = y.argsort()[-top_n:]
for idx in top_indices:
    bars[idx].set_color('orange')
    plt.text(x[idx], y[idx] + 2, str(x[idx]), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title("Autocorrelation Peaks for Vigenère Key Length Detection")
plt.xlabel("Shift (Possible Key Length)")
plt.ylabel("Match Count")
plt.xticks(range(1, max_shift + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

english_freq = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def chi_squared_score(text_segment):
    N = len(text_segment)
    if N == 0:
        return float('inf')
    obs_counts = Counter(text_segment)
    chi2 = 0.0
    for ch in alphabet:
        observed = obs_counts.get(ch, 0)
        expected = english_freq[ch] * N
        if expected < 1e-6: expected = 1e-6
        chi2 += (observed - expected) ** 2 / expected
    return chi2

def shift_decrypt(text_segment, shift):
    return ''.join(chr((ord(c) - 65 - shift) % 26 + 65) for c in text_segment)

def find_key_chi2(ciphertext, key_len):
    key_shifts, key_letters, scores = [], [], []
    for i in range(key_len):
        col = ciphertext[i::key_len]
        best_shift, best_score = None, float('inf')
        for s in range(26):
            dec = shift_decrypt(col, s)
            score = chi_squared_score(dec)
            if score < best_score:
                best_score, best_shift = score, s
        key_shifts.append(best_shift)
        key_letters.append(chr(65 + best_shift))
        scores.append(best_score)
    return ''.join(key_letters), key_shifts, scores

def vigenere_decrypt(ciphertext, key):
    out = []
    for i, c in enumerate(ciphertext):
        k = ord(key[i % len(key)]) - 65
        out.append(chr((ord(c) - 65 - k) % 26 + 65))
    return ''.join(out)

ciphertext = re.sub('[^A-Z]', '', ciphertext_raw.upper())
key_len = 5
key, shifts, scores = find_key_chi2(ciphertext, key_len)
plaintext = vigenere_decrypt(ciphertext, key)

print(f"\nRecovered key (length={key_len}): {key}")
print("Shifts:", shifts)
print("Per-column chi2:", ["{:.2f}".format(s) for s in scores])
print("\nDecrypted start:\n", plaintext[:400])
print("\nAverage chi2 per column for key lengths 1..12:")
for L in range(1, 13):
    _, _, scores = find_key_chi2(ciphertext, L)
    avg = sum(scores) / L
    print(f"len={L:2d} -> avg chi2 = {avg:.2f}")
