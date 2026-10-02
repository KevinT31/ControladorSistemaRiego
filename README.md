<div align="center">

# Intelligent Irrigation Controller

### IoT · Machine Learning · Automation · Cloud Integration

</div>

---

## Overview

Python-based smart-irrigation system that coordinates **sensor acquisition, signal conditioning, decision logic, actuator control, model training and cloud-oriented utilities**.

This is the broader of two related irrigation repositories in this profile. The compact prototype is available in [Siemens](https://github.com/KevinT31/Siemens).

## Architecture

~~~mermaid
flowchart LR
    Sensors[Sensors] --> Conditioning[Signal Conditioning]
    Conditioning --> Decision[Decision Engine]
    Decision --> Controller[Central Controller]
    Controller --> Actuators[Pump / Valves]

    Training[Model Training] --> Decision
    Synthetic[Synthetic Data] --> Training
    Cloud[Cloud Sync / Cloud Training] --> Decision
~~~

## Main Components

- environmental/hydraulic sensor abstraction
- signal conditioning
- central controller
- machine-learning-assisted decisions
- rule-based fallback logic
- pump/valve actuator control
- local model training
- cloud-oriented training utilities
- cloud synchronization
- synthetic data generation
- GUI
- test modules

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Flask · Google Cloud libraries · Modbus · Adafruit libraries · Matplotlib

## Structure

~~~text
src/
├── main.py
├── controller.py
├── sensors.py
├── signal_conditioning.py
├── decision_engine.py
├── actuators.py
├── model_training.py
├── cloud_model_training.py
├── cloud_sync.py
├── generate_synthetic_data.py
├── gui.py
└── tests/
~~~

## Run Locally

~~~bash
git clone https://github.com/KevinT31/ControladorSistemaRiego.git
cd ControladorSistemaRiego

python -m venv .venv
# activate the virtual environment

pip install -r requirements.txt
python src/main.py
~~~

## Engineering Notes

### Fallback control

Irrigation logic should not depend exclusively on a serialized ML model. The broader architecture preserves deterministic/rule-based decision paths.

### Training vs runtime

Training utilities are separated from the runtime controller loop so experimentation does not become a runtime dependency.

### Synthetic data

Synthetic-data generation is explicit and should be understood as engineering/testing support rather than real field telemetry.

## Scope

This repository is an engineering prototype for IoT/automation architecture. It should not be interpreted as a certified agricultural control system or a production hardware deployment.

---

### What this project demonstrates

**Python modularity · IoT control loops · ML-assisted automation · actuator/sensor integration · cloud-oriented workflows**
