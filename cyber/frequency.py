import matplotlib.pyplot as plt
import numpy as np

def frequency(text, max_shift=100):
    matches = {}
    for move in range(1, min(max_shift, len(text))):
        count = 0
        for i in range(len(text) - move):
            if text[i] == text[i + move]:
                count += 1
        matches[move] = count
    return matches

def plot_autocorrelation(matches, max_shift=50):
    x = np.array(list(matches.keys()))
    y = np.array(list(matches.values()))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, y, color='skyblue', edgecolor='k')
    top_n = 10
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


if __name__ == '__main__':
    text = '''UEKQG VFVMF ODMFT KIMIM YRLQM XCQFF RVOUD RNPAI GJTAH KUJKQ BVZKA TVETA RFWWQ JRBTQ XSCFY UJBAR GCTNK NVZSD GELYA ZYMDM TUBTQ XVEME TFBTU TXBTM ZJPQI ULTPZ UKPMH KXQHQ TKWFT KTPUX JFVOQ YYMSM BVPQD GCQFF RVZUP OEOTA UUWRD KUDQX BVBIT OTPEG OKMPT KIAAI KCTFT GKATQ CFCXP TVDQD CVIDM TPBTU TXMXE KJWET KNIEM RNIKE IRTXQ JCQFF RVZQP XZLUZ MYWAP UEMPM EYMDY UKPQD YRQPF UYMDO UDMXU ZKTQD KUZUP OEOTA UUPQD KZAMB OVKQA LTIWQ GELMN UKBXQ UWEUZ KKIWQ ZYMYF UPWGD MIIZP SFBTQ XJPQU YZTXM TUEQM QRVPF NVGIU RCLAT KIOAA JJMFA AKJQR UIMUF MVBET UKIZP CYMZK ULIDQ MFQZS CRTWZ OTMXK GELCG OVBXK GELPA TFBDG TFNRF NVXMF NFZKA ADIKR GCTMZ JSZQM QKPQN UKBXQ GELFT KEGAG XXZMZ JDWFT KIEUX RXMFZ UKPUZ MRVPI NVVKA AXWUZ ZFPQD XFWYP UEBRA XXMFF UJIKS UFLYA XEQZS GELPA TKXQQ VZVFA KMMDK IFZZQ XSMRA XVGAG JFQFU CZTXF GBMSD KRBOM XVAMU JCQFF RVZQP XZLUZ MYWAP ZFPQD SFBTQ XRVPS GMMTQ XYIZP UEQFF NVODM TUUAF NVZXU BVLAG ZZVFT KNWAP NRTRM RVISG KWZAY ZYMHU RCISQ GELVG YKIEX OKBXQ XVLDU JZVST UFLQZ ZVZQP ZYMIA UUIIA RWUQF NVZDQ JIQPU TXPAA JUQPZ UKSZA CNPMF GNQOW KUKDQ GKCDQ NVEME GELIM YEWFM ZRTXM LIIUP UWPUY MFWPP GPTUF ZCMDQ JIQPU TXPAA JJIUP NVBTM TBGAG QZVPX ENWXR CYQFT KIIIM EJWQM XCGXU ZKTQD KUZUP OEOTA UUBAY EXZMZ JDWFT KIAIT GKPMH KPWGS UKQZK ULZMB XFVOM QVIZP CZVQK KJBQD JRGIM YSIWU TXLMK YFXAA XJQOW MIIZP SFBTQ XZAFA NRDQE UDMFT OEOSA UUBAY GBMTQ XJBDA TXMDI NVZQP UVAKA AIODM TUUAF NVZXU BVTUF ZCMDQ JIQPU TXPAA JROAA JHCMD ZVZAR GCMMS AVNMD ZYMDA TZVFT KNWAP NVZTA AJMEF GELEG TUMDF NVBTD KVTMD MVWMW ZIMQE ZYMZG ZKZQQ YRZQV AJBNQ RFEKA AJCDQ RPUGE ZBVAI OKZQB RZMPX OKBXQ XVLDU JZVST UFL'''
    clean_text = ''.join(text.split())
    max_shift = 50
    autocorrelation_matches = frequency(clean_text, max_shift=max_shift)
    plot_autocorrelation(autocorrelation_matches, max_shift=max_shift)
