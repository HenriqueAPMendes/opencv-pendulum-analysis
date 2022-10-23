import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#define equacao de OHA para ajustar o fit
def oha(t, a, b, w, p, l):
    return a*np.exp(-b*t)*np.cos(w*t - p) + l


cap = cv2.VideoCapture("Video2 Pendulo.mp4")

ok, frame = cap.read()
if not ok:
    print("Não foi possível ler o arquivo")
    exit(1)

coordenadas = []
tempo = []

bbox = cv2.selectROI('Tracker', frame)
cv2.destroyAllWindows()

tracker = cv2.legacy.TrackerCSRT_create()
multitracker = cv2.legacy.MultiTracker_create()

multitracker.add(tracker, frame, bbox)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    ok , boxes = multitracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), 1, 1)
        
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t > 0.1:
            tempo.append(t/1000)
            coordenadas.append((x+w/2)/100)

    cv2.imshow('MultiTracker', frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

min = min(coordenadas)
max = max(coordenadas)

#A value guess
guessA = (max-min)/2

fig = plt.figure()
plt.title('Gráfico Pêndulo Físico')
plt.ylabel('Posição (pixels/100)')
plt.xlabel('Tempo(s)')


tempo = np.array(tempo)
coordenadas = np.array(coordenadas)
initialGuess = [guessA, 0.1, 5, 0, 3]

values = curve_fit(oha, tempo, coordenadas, initialGuess)
a = values[0][0]
b = values[0][1]
w = values[0][2]
p = values[0][3]
l = values[0][4]

#Calculo fator de qualidade
#Válido para valores pequenos de b

t = 2*np.pi / w # periodo

q = 2*np.pi*(1/(1-np.exp(-2*b*t)))

plt.plot(tempo, oha(tempo, a, b, w, p, l) - l)
plt.plot(tempo, coordenadas - l, "bo", markersize=2, color='red') #scatter
print('fit params: a = {:.2f}, b = {:.2f}, w = {:.2f} p = {:.2f}\n'.format(a, b, w, p, l))
print("FATOR DE QUALIDADE = {}.\n".format(q))
plt.show()
 
