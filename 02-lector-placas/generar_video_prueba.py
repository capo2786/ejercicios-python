import cv2, numpy as np
W,H,FPS,SEC = 800,450,25,24
out = cv2.VideoWriter("autos.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
rng = np.random.default_rng(3)
placas = ["PBA1234","GSC7789","ABG4521"]
for i in range(FPS*SEC):
    t = i/FPS
    f = np.full((H,W,3), 95, np.uint8)
    cv2.rectangle(f,(0,0),(W,150),(140,150,150),-1)      # fondo
    cv2.rectangle(f,(0,330),(W,H),(70,70,72),-1)          # asfalto
    k = int(t//8)                                          # un auto cada 8 s
    if k < len(placas):
        u = (t - k*8)/8.0
        cx = int(-150 + u*(W+300)); cy = 300
        cv2.rectangle(f,(cx-120,cy-90),(cx+120,cy+80),(60,55,120),-1)  # carroceria
        pw,ph = 150,46
        x0,y0 = cx-pw//2, cy+10
        cv2.rectangle(f,(x0,y0),(x0+pw,y0+ph),(250,250,250),-1)
        cv2.rectangle(f,(x0,y0),(x0+pw,y0+ph),(30,30,30),2)
        cv2.putText(f,placas[k],(x0+9,y0+34),cv2.FONT_HERSHEY_SIMPLEX,0.78,(20,20,20),2,cv2.LINE_AA)
    f = np.clip(f + rng.normal(0,3,(H,W,3)),0,255).astype(np.uint8)
    out.write(f)
out.release(); print("autos.mp4 listo con placas:", placas)
