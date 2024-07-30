from values import delta_x, delta_y, delta_t, data
import numpy as np
import matplotlib.pyplot as plt

def attenuation_compute(trace, delta_t):
    lst=[]
    i=0
    for value1, value2 in zip(trace[:-1], trace[1:]):

        if value1!=0:
            if value2/value1>0:
                lst.append(value2/value1)

        i+=1
    log1=np.log(lst)
    return -1*np.mean(log1)/delta_t

def f_k_plot(delta_t, y, data, delta_x):
    F=data[:,y,:]
    m, n = F.shape
    df = 1 / n / delta_t
    dk = 1 / m / delta_x
    f=np.arange(-n/2, n/2)*df
    k=np.arange(-m/2, m/2)*dk
    k=k[int((k.shape[0]+1)/2):]
    f=f[int((f.shape[0]+1)/2):]
    k = k * 2 * np.pi
    st = np.fft.fftshift(np.fft.fft2(F)) * delta_t*delta_x
    st = np.flip(st, axis=1)
    spec = np.abs(st)
    spec = spec[int((spec.shape[0] + 1) / 2):, int((spec.shape[1] + 1) / 2):]
    fig, ax0 = plt.subplots()
    im = ax0.pcolormesh(f, k, np.abs(spec), shading="gouraud")
    plt.title(f"f-k plot at the point y={y}")
    plt.ylabel('Wavenumber (rad/m)')
    plt.xlabel('Frequency (Hz)')
    plt.ylim([0, 0.05])
    plt.xlim([0, 35])
    fig.colorbar(im, ax=ax0)
    plt.savefig(f"f-kplot_{y}.pdf")
    plt.close()

def seismic_cube_ft(data):
    trace_num = data.shape[0] * data.shape[1]
    n=data.shape[2]
    amplitudes=np.zeros(int(n/2+1))
    for v in range(trace_num):
        i = int(v / data.shape[1])
        j = int(v % data.shape[1])
        amplitudes+=np.abs(np.fft.rfft(data[i, j, :]))

    return amplitudes/trace_num/np.sqrt(amplitudes.shape[0])

def signaltonoise_dB(a):
    m = a.mean()
    sd = a.std()
    return 10*np.log10(abs(np.where(sd == 0, 0, m/sd)))