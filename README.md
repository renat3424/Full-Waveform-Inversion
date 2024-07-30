# Full Waveform Inversion
## Analysis of seismic traces from Full Waveform Inversion Data



This is a program that performs basic analysis of full waveform inversion data such as calculation of avarage fourier transform of traces, attenuation rate, stft with different windows, frequency-wavenumber plot, signal to noice ratio and so on.  Model is trying to simulate a behaviour of amplitude traces as prediction of timeseries with use of lstm architecture. The data is represented by seismic cube of dimensions x, y, t, where x is a measurement axis, delta_x  (distance between one point to another) is equal to 10.5 meters, same for y, delta_t is equal to 4 milliseconds. The results of program's work can be found in presentation. The data can be found at https://drive.google.com/file/d/1E5QCkFK7WGJGE0yQo8nt7OsUdwKuTG4i/view?usp=sharing as numpy array with dimensions x, y, t.  

