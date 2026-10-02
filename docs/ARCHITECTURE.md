# Intelligent Irrigation Controller — Architecture

## Runtime Control Loop

~~~text
Sensors
   ↓
Signal conditioning
   ↓
Decision engine
   ↓
Central controller
   ↓
Actuators
~~~

## Supporting Flows

- local training → decision engine
- cloud-oriented training → model artifacts
- synthetic-data generation → training/testing
- cloud synchronization ↔ operational data

## Design Goal

Keep the physical-control path independent from model-training and cloud utilities so the controller can retain deterministic fallback behavior.
