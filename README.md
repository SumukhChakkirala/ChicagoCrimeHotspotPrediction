# 🚔 Chicago Crime Hotspot Prediction

A spatio-temporal crime risk prediction system for the city of Chicago, built using **Graph Convolutional Networks (GCN)** and visualized through an **interactive React web app**.

---

## 📌 Overview

This project predicts crime hotspots in Chicago using historical crime data and machine learning techniques. The system processes crime records, builds a graph-based representation of locations, and uses a Graph Convolutional Network (GCN) to identify high-risk areas.

---

## 🎯 Objectives

* Analyze historical crime data
* Identify spatial patterns of crime
* Predict future crime hotspots
* Visualize results using maps and heatmaps

---

## 🛠 Tech Stack

* Python (Pandas, NumPy, Scikit-learn)
* PyTorch & PyTorch Geometric (GCN)
* FastAPI (Backend API)
* React.js + Leaflet (Frontend visualization)
* Node.js

---

## 🏗️ Project Structure

```
project/
├── app.py              ← FastAPI backend (REST API)
├── model.py            ← GCN model training
├── preprocess.py       ← Data preprocessing pipeline
├── crime-map/          ← React frontend
│   ├── src/
│   │   ├── App.js      ← Main React component
│   │   └── App.css     ← Styles
│   └── public/
│       └── index.html
├── processed/          ← Generated after preprocessing (not in repo)
└── outputs/            ← Generated after training (not in repo)
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description                                   |
| -------- | ------ | --------------------------------------------- |
| /predict | GET    | Returns crime risk for selected location/time |
| /nodes   | GET    | Returns all grid nodes                        |
| /types   | GET    | Returns available crime types                 |

---

## Demo

* Interactive map of Chicago with color-coded crime risk per grid cell
* Select a **date and time** to get temporally adjusted predictions
* Click any location to see a full crime risk breakdown
* Filter by crime type (THEFT, BATTERY, NARCOTICS, etc.)

---

## Requirements

* Python 3.11
* Node.js 18+
* ~2 GB disk space for the dataset

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/PramitiTD/ChicagoCrimeHotspotPrediction.git
cd ChicagoCrimeHotspotPrediction
```

---

## Step 2 — Install Python dependencies

```bash
pip install pandas numpy scikit-learn scipy tqdm fastapi uvicorn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

---

## Step 3 — Download the dataset

1. Go to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
2. Click **Export → CSV**
3. Wait for the full file to download (~1.7 GB, ~8 million records)
4. Rename it to `chicago_crimes.csv` and place it in the project root

---

## Step 4 — Run preprocessing

```bash
python preprocess.py
```

This will create the `processed/` folder with:

* `crimes_processed.csv`
* `nodes.csv`
* `edges.csv`
* `label_map.csv`

Expected output:

```
Records  : ~8,000,000
Nodes    : ~300-400
Edges    : ~2000+
```

---

## Step 5 — Train the GCN model

```bash
python model.py
```

This will create the `outputs/` folder with:

* `gcn_model.pt`
* `node_predictions.csv`
* `training_curves.png`
* `hotspot_heatmaps.png`

Expected output:

```
Test Accuracy : ~85%
Macro F1 Score: ~0.55
```

---

## Step 6 — Install frontend dependencies

```bash
cd crime-map
npm install
cd ..
```

---

## Step 7 — Run the project

Open **two terminals**:

**Terminal 1 — Backend**

```bash
cd ChicagoCrimeHotspotPrediction
uvicorn app:app --reload --port 8000
```

Wait until you see:

```
✓ Loaded 327 nodes with 9 crime types
✓ Temporal index ready
INFO: Application startup complete.
```

**Terminal 2 — Frontend**

```bash
cd ChicagoCrimeHotspotPrediction/crime-map
npm start
```

Then open your browser and go to:

```
http://localhost:3000
```

---

## How to use the app

| Action                | Result                                            |
| --------------------- | ------------------------------------------------- |
| Select a date         | Changes temporal prediction context               |
| Select a time         | Adjusts risk based on hour of day                 |
| Select crime type     | Filters map to show risk for that crime only      |
| Click on the map      | Shows crime risk breakdown for that location      |
| Click outside Chicago | Shows an error message (data only covers Chicago) |

---

## How it works

1. **Preprocessing** — 8M crime records are aggregated into 1.5km × 1.5km grid cells across Chicago
2. **Graph construction** — Grid cells become nodes; cells within 2.5km are connected by edges weighted by inverse distance
3. **GCN training** — A 2-layer Graph Convolutional Network learns spatial crime patterns from node features and graph structure
4. **Temporal adjustment** — At prediction time, base GCN scores are adjusted using historical hourly/daily/monthly crime patterns
5. **Visualization** — Risk scores are served via FastAPI and displayed on an interactive Leaflet map

---

## Crime Types

| Type                | Description                     |
| ------------------- | ------------------------------- |
| THEFT               | Stealing property without force |
| BATTERY             | Physical assault                |
| CRIMINAL DAMAGE     | Vandalism, property destruction |
| NARCOTICS           | Drug-related offences           |
| BURGLARY            | Breaking into buildings         |
| MOTOR VEHICLE THEFT | Car theft                       |
| DECEPTIVE PRACTICE  | Fraud, scams                    |
| ASSAULT             | Threatening behaviour           |
| OTHER OFFENSE       | Miscellaneous crimes            |

---

## Dataset

**Chicago Crimes - 2001 to Present**
Source: Chicago Data Portal (Chicago Police Department)
URL: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
Size: ~8 million records, ~1.7 GB

> The dataset is not included in this repository due to its size. Download it separately using the link above.

---

## ⚠️ Limitations

* Predictions depend heavily on historical data
* Does not account for real-time events
* Spatial resolution limited to grid size
* Model performance varies across crime types

---

## 🤝 Contributors

* CKD Sumukh
* Pramiti T D

---

## 📚 References

* Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017
* Chicago Crime Dataset — Chicago Data Portal
* PyTorch Geometric — https://pytorch-geometric.readthedocs.io
