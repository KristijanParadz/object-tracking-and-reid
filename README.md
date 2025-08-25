# Multi-Camera Object Tracking and Re-Identification

This repository contains the code developed as part of my **Master’s degree project**.  
It implements **object tracking and re-identification across multiple cameras** using **YOLO** for detection and **ResNet-50** for feature extraction.

---

## 🚀 Features
- Multi-camera **object tracking** and **re-identification**  
- **YOLO** for detection  
- **ResNet-50** for feature embedding  
- **CUDA** support (if available)  
- **Dockerized setup** for easy deployment  
- **FastAPI** backend with a **Vue.js** frontend  
- Backend and frontend communicate using **Socket.IO**  

---

## ⚙️ Prerequisites
- **Linux** (project is configured for Linux usage)  
- **Docker** and **Docker Compose** installed  

Before running the system, make sure to review and configure:  
- [`config.py`](./backend/frame_processing/config.py)  
- [`docker-compose.yml`](./docker-compose.yml)  

---

## 🐳 Running with Docker
To start the system, run:

```bash
docker compose up -d
```
This will spin up the FastAPI backend and Vue frontend.

---

## 🎥 Camera Calibration

Before starting the algorithm, you must **calibrate your cameras**.  
The system relies on calibration data to correctly perform multi-camera tracking.

If you only want to **see how the system works without real calibration**, you can insert fake calibration data into your browser’s `localStorage`.

Example data (add manually in your browser’s developer console):

```javascript
localStorage.setItem("calibrationData", JSON.stringify([
  {
    "index": 0,
    "K": [[437.244832361936,0,280.6146406757888],[0,407.0888116954003,221.31406638599236],[0,0,1]],
    "distCoef": [0.49105131329891605,-0.999738966093766,-0.020394698564438166,-0.02429188856837534,0.9070218199689312],
    "R": [[1,0,0],[0,1,0],[0,0,1]],
    "t": [[0],[0],[0]]
  },
  {
    "index": 1,
    "K": [[2447.037883469437,0,409.32705161914555],[0,2677.231962920113,231.14361280477058],[0,0,1]],
    "distCoef": [6.783393553690082,-1635.6039622764768,0.13465903010102132,-0.3425366394755081,61334.710687960476],
    "R": [[0.9742210500721973,0.10115023847668242,0.20164814616637466],[0.05590369330216974,0.7577249261507168,-0.6501751405314322],[-0.21855919717871009,0.6446871842576065,0.7325369013111878]],
    "t": [[-4.26197674433809],[3.446971583806989],[5.440394268493307]]
  },
  {
    "index": 2,
    "K": [[1200,0,640],[0,1250,360],[0,0,1]],
    "distCoef": [0.1,-0.2,0.01,0.01,0.05],
    "R": [[0.99,0.01,0.02],[0.01,0.98,-0.15],[-0.02,0.15,0.97]],
    "t": [[2],[1],[3]]
  },
  {
    "index": 3,
    "K": [[1300,0,640],[0,1350,360],[0,0,1]],
    "distCoef": [0.2,-0.3,0.02,0.02,0.1],
    "R": [[0.95,-0.1,0.25],[0.05,0.98,0.15],[-0.2,-0.1,0.95]],
    "t": [[-2],[2],[4]]
  }
]));
```

## 📡 System Architecture
- **Backend**: FastAPI  
- **Frontend**: Vue.js  
- **Communication**: Socket.IO  

---

## 📝 Notes
- System is configured for **Linux** environments.  
- Check `config.py` and `docker-compose.yml` before running.  
- CUDA will be used automatically if available.  
