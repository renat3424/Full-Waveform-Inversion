import numpy as np
from values import delta_x, delta_y, delta_t, data
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import signal

trace=data[400,400,:]

fs=1/delta_t
rectangular = signal.windows.hamming(32)
hann=signal.windows.hann(32)
flat=signal.windows.flattop(32)
dic={"rectangular": rectangular, "hann": hann, "flat-top": flat}

f, t, Zxx=signal.stft(trace, window=rectangular, fs=fs, nperseg=32)
plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
plt.show()
time1=0
for i in range(5):
    time1=15+i*2
    plt.figure(figsize=(12, 6))
    plt.title(f'Frame at time {t[time1]}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('amplitude')
    lines=[]
    for key, value in zip(dic.keys(), dic.values()):
        f, t, Zxx=signal.stft(trace, window=value, fs=fs, nperseg=32)
        line = plt.plot(f,np.abs(Zxx[:, time1]))
        line[0].set_label(key)
        lines.append(line[0].get_label())
    plt.legend(lines)
    plt.show()




