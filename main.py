import numpy as np
from values import delta_x, delta_y, delta_t, data
import matplotlib.pyplot as plt
from utils import seismic_cube_ft, attenuation_compute, f_k_plot, signaltonoise_dB

if __name__=="__main__":
    freq = np.fft.rfftfreq(data.shape[2], d=delta_t)

    avarage_ft = seismic_cube_ft(data)
    plt.figure(figsize=(12, 6))
    plt.title("Averaged FT spectrum")
    plt.xlabel("frequency(Hz)")
    plt.ylabel("amplitude")
    plt.plot(freq, avarage_ft, "r", linewidth=2)
    plt.savefig("averaged_ft_real1.pdf")
    plt.close()

    print("attenuation rate of trace at points x=0, y=0", attenuation_compute(data[0, 0], delta_t))
    print("attenuation rate of trace at points x=500, y=500", attenuation_compute(data[500, 500], delta_t))
    print("attenuation rate of trace at points x=200, y=300", attenuation_compute(data[200, 300], delta_t))
    print("attenuation rate of trace at points x=100, y=500", attenuation_compute(data[500, 100], delta_t))
    print("attenuation rate of trace at points x=189, y=450", attenuation_compute(data[189, 450], delta_t))

    f_k_plot(delta_t, 0, data, delta_x)
    f_k_plot(delta_t, 200, data, delta_x)
    f_k_plot(delta_t, 600, data, delta_x)

    n = data.shape[2]
    n1 = data.shape[1]
    n2 = data.shape[0]
    print("snr at entire cube: ", signaltonoise_dB(data))
    print("snr at point t=0: ", signaltonoise_dB(data[:, :, 0]))
    print(f"snr at point t={int(n / 2)}: ", signaltonoise_dB(data[:, :, int(n / 2)]))
    print(f"snr at point t={int(n - 1)}: ", signaltonoise_dB(data[:, :, int(n - 1)]))
    print("snr at point y=0: ", signaltonoise_dB(data[:, 0, :]))
    print(f"snr at point y={int(n1 / 2)}: ", signaltonoise_dB(data[:, int(n1 / 2), :]))
    print(f"snr at point y={int(n1 - 1)}: ", signaltonoise_dB(data[:, int(n1 - 1), :]))
    print("snr at point x=0: ", signaltonoise_dB(data[0, :, :]))
    print(f"snr at point x={int(n2 / 2)}: ", signaltonoise_dB(data[int(n2 / 2), :, :]))
    print(f"snr at point x={int(n2 - 1)}: ", signaltonoise_dB(data[int(n2 - 1), :, :]))