JAIR CODE
===

This is a refined version of Tensorflow planner on planning problem. 

In the training stage, we train the transition functions through the previous observations.
In another words, we assume the trainsition function is unknown while the reward function is given.

The code is able to connect to the RDDL simulator by calling python through commandline tools.


# Example
Train
```bash
python train.py \
-p domains/res/reservoir4/ \
-x Reservoir_Data.txt \
-y Reservoir_Label.txt \
-w weights/reservoir/reservoir4 \
-s 4 \
-d Reservoir
```

Plan
```bash
python plan.py \
-w weights/reservoir/reservoir3 \
-d Reservoir \
-i Reservoir3 \
-s 3 \
-a 3 \
--initial temp/state
```

Note: the initial state is optional, and the default is zero state.


