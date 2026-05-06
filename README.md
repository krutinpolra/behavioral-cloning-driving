# Self-Driving Car Simulation Using CNN

**Members:** Ikechukwu Charles Okorji, Krutin Bharatbhai Polra

A behavioral cloning project that trains a convolutional neural network (based on the NVIDIA architecture) to predict steering angles from front-camera images, enabling a simulated car to drive itself around a track.

---

## Project Structure

```
├── train.py                   # Data loading, augmentation, preprocessing, model training
├── TestSimulation.py          # Inference server — connects trained model to the simulator
├── model.h5                   # Saved trained model weights (best validation loss)
├── driving_log.csv            # Recorded driving data (image paths + steering angles)
├── IMG/                       # Front-camera images collected from the simulator
├── steering_distribution.png  # Histogram of steering angle distribution in the dataset
├── training_history.png       # Training vs. validation loss curves
├── package_list.txt           # Conda environment package list
└── Final_Project.pdf          # Original project brief
```

---

## Getting Started

### 1. Download the Simulator

Download and extract the Udacity Self-Driving Car Simulator:  
https://github.com/udacity/self-driving-car-sim

The simulator is used in two stages: **Training Mode** to collect driving data, and **Autonomous Mode** to test the trained model.

### 2. Set Up the Environment

This project requires **Python 3.8** and **TensorFlow GPU 2.3** (tested on Windows 10/11, 64-bit).

```bash
conda create --name cvi620 --file package_list.txt
conda activate cvi620
```

#### Key dependencies

| Package | Version |
|---|---|
| Python | 3.8 |
| TensorFlow GPU | 2.3.0 |
| OpenCV | included via conda |
| Flask + Flask-SocketIO | 1.1.2 / 3.3.1 |
| eventlet | 0.25.1 |
| pandas | 1.2.4 |
| scikit-learn | 0.24.2 |
| Pillow | via conda |

---

## Step 1 — Collect Training Data

1. Launch `beta_simulator.exe` (640×480, Fastest quality, Windowed)
2. Select **Training Mode** on Track 1 (leftmost track)
3. Click **Recording** and choose your project folder as the save path
4. Drive ~5 laps forward and ~5 laps in reverse using mouse steering for smooth inputs
5. Click **Recording** again to stop — this generates:
   - `IMG/` — center, left, and right camera images
   - `driving_log.csv` — columns: `center, left, right, steering, throttle, brake, speed`

Only the **center camera** images and **steering** values are used for training.

---

## Step 2 — Train the Model

With the conda environment active, run:

```bash
python train.py
```

What `train.py` does:
1. Loads `driving_log.csv` and fixes simulator absolute paths to local `IMG/` paths
2. Plots and saves a steering angle distribution histogram (`steering_distribution.png`)
3. Splits data 80/20 into training and validation sets
4. Applies **data augmentation** to the training set only:
   - Random horizontal flip (steering angle negated)
   - Random brightness adjustment
   - Random panning (horizontal + vertical)
   - Random zoom
   - Random small rotation (±5°)
5. Preprocesses every image:
   - Crop rows 60–135 (removes sky and hood, keeps road)
   - Convert RGB → YUV
   - Apply Gaussian blur (3×3)
   - Resize to 200×66 (NVIDIA model input)
   - Normalize pixel values to [0, 1]
6. Trains the **NVIDIA CNN architecture**:
   - 5 convolutional layers (5×5 and 3×3 kernels, ELU activations)
   - Flatten
   - 3 fully-connected layers (100 → 50 → 10 neurons, ELU)
   - 1 output neuron (steering angle)
   - Optimizer: Adam (lr=1e-3), Loss: MSE
7. Saves the best model to `model.h5` (monitored by validation loss)
8. Saves training/validation loss curves to `training_history.png`

Training configuration: **batch size = 32, epochs = 10**

---

## Step 3 — Test the Model in the Simulator

1. Activate the conda environment:
   ```bash
   conda activate cvi620
   ```

2. Start the inference server:
   ```bash
   python TestSimulation.py
   ```
   This loads `model.h5` and starts a SocketIO server on port **4567**.

3. Launch `beta_simulator.exe` with the same settings used during data collection.

4. Select **Autonomous Mode** — the car will connect to the server and drive itself.

The server receives real-time front-camera frames from the simulator, preprocesses each frame identically to training (crop → YUV → blur → resize → normalize), predicts a steering angle, and sends back steering + throttle commands. The car's speed is capped at **10 mph**.

---

## Results

- **Steering distribution:** The dataset shows a concentration of near-zero angles (straight driving) with a spread across left/right turns — consistent with a full track recording in both directions.
- **Training curves:** Loss decreased steadily over 10 epochs with no significant overfitting.
- **Simulator performance:** The trained model successfully keeps the car on the road for a complete lap on Track 1.

---

## Approach & Challenges

**Approach:**  
We followed the NVIDIA end-to-end self-driving CNN architecture, which directly maps raw camera pixels to steering commands. Data augmentation was critical to prevent the model from overfitting to the straight-road bias in the training data.

**Challenges:**
- **Path mismatch:** The simulator saves absolute paths in `driving_log.csv`. These had to be remapped to local relative `IMG/` paths before training.
- **Dataset imbalance:** The raw data is heavily skewed toward zero steering (straight driving). Driving in both directions and applying random flipping during augmentation helped balance the distribution.
- **Library compatibility:** The `TestSimulation.py` server requires specific versions of `flask-socketio`, `python-socketio`, and `eventlet` that are pinned in `package_list.txt`. Using the exact conda environment is essential to avoid connection issues with the simulator.
