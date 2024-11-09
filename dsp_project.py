import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import queue

fs = 16000 #sampling freq
dur = 0.05 #frame duration in sec
spd = 343 #speed of sound 
d = 0.1 #distance b/t microphones
m = 4 #no.of micphs
frm_size = int(fs * dur) #frame size

q = queue.Queue()

def cb(indata, frm, time, stat):
    q.put(indata.copy())

st = sd.InputStream(samplerate=fs, channels=m, callback=cb)
st.start()

while True:
    if q.qsize() < 1:
        continue

    data = q.get()
    
    delys = []
    for i in range(1, m):
        x_corr = correlate(data[:, 0], data[:, i], mode='full')
        max_idx = np.argmax(np.abs(x_corr))
        delay = max_idx - (len(data[:, 0]) - 1)
        delys.append(delay)

    t_delays = np.array(delys) / fs * spd
    angs = np.degrees(np.arcsin(t_delays / d))
    print("Estimated Angles: ",angs)

    shifted = np.zeros_like(data)
    for i in range(m):
        if i > 0:
            delay_samples = int(delys[i - 1])  
        else :
            0
        shifted[:, i] = np.roll(data[:, i], delay_samples)
    
    bf = np.sum(shifted, axis=1) / m
    bf /= np.max(np.abs(bf))
    
    print("Beamformed Signal Frame:", bf)
