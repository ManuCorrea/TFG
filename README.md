# KinderGarten

## Abstract
Robotics is going to vastly shape our world in the following years. To achieve that, we need to develop and teach our robots to perform the desired tasks while having expected and desirable behaviours.

The development process of robotics simulation and Reinforcement Learning is currently experiencing a wave of innovation. Due to the complexity of the real world and the complexity of generalising the task to every possible situation, the development process of a robotics solution can be challenging.

In order to aid in the process of creating the robotics solutions, KinderGarten was created to provide a set of tools to aid in the development process of robotics simulations and training with Reinforcement Learning.

## Instalation
```
conda create -n kg python=3.7
conda activate kg
cd kinder-garten
python -m pip install -e .
```
All example scripts are under kinder-garten

## Editor
```
python editor.py
```

## Training with script
```
python ejemplo_kg.py
```

## Training with GUI
```
python train_gui.py
```

## Show results of training session
```
python ejemplo_kg_show.py
```

## For simple visualization
Agent performing random actions
```
python ejemplo_kg_vis_simple.py
```

# Controlling the gripper environment with an xbox controller
```
python ejemplo_kg_controller.py
```
