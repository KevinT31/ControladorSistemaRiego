# Intelligent Irrigation Controller

Modular smart-irrigation system with sensor processing, machine learning, cloud synchronization and actuator control.

## Overview

This repository contains a Python-based irrigation controller designed around a modular architecture. The system coordinates sensors, signal conditioning, decision logic, actuators and model-training utilities through a central controller.

The project includes both local machine-learning workflows and cloud-oriented modules, together with synthetic-data generation and tests for core components.

## Main Features

- Environmental and hydraulic sensor acquisition
- Signal conditioning and validation
- Automatic irrigation control
- Machine-learning-assisted decision engine
- Rule-based fallback decisions
- Pump and valve actuator control
- Local and cloud-oriented model training
- Cloud synchronization utilities
- Synthetic data generation
- GUI and test modules

## Tech Stack

- Python
- pandas / NumPy
- scikit-learn
- XGBoost
- Google Cloud libraries
- Flask
- Modbus
- Adafruit sensor libraries
- Matplotlib

## Project Structure

```text
src/
├── main.py                    # Application entry point
├── controller.py              # Main system orchestration
├── sensors.py                 # Sensor acquisition
├── signal_conditioning.py     # Signal preprocessing
├── decision_engine.py         # Decision logic
├── actuators.py               # Pump/valve control
├── model_training.py          # Local model training
├── cloud_model_training.py    # Cloud-oriented training workflow
├── cloud_sync.py              # Cloud synchronization
├── generate_synthetic_data.py
├── gui.py
└── tests/                     # Additional tests
```

## Getting Started

### Requirements

- Python 3
- A virtual environment is recommended

### Installation

```bash
git clone https://github.com/KevinT31/ControladorSistemaRiego.git
cd ControladorSistemaRiego
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python src/main.py
```

## Architecture

At a high level, the project follows this flow:

```text
Sensors
   ↓
Signal conditioning
   ↓
Decision engine
   ↓
Central controller
   ↓
Actuators
```

Cloud synchronization and model-training modules complement the local control loop.

## Portfolio Notes

This project demonstrates Python software architecture for an IoT/automation scenario, including sensor integration, control logic, machine learning, cloud integration and testing.
