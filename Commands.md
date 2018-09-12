Commands
===
The following commands are used in generating experiment results on our paper.

## Training
We train 3 domains in the paper: Reservoir, HVAC, Navigation.

Reservoir : 1 hidden layer, 32 neurons in layer, densely connected
HVAC: 1 hidden layer, 32 neurons in layer, densely connected
Navigation: 2 hidden layers, 32 neurons in each layer, densely connected

### Reservoir 3
```bash
python train.py -p data/Reservoir/Reservoir_3/ -x Reservoir_Data.txt -y Reservoir_Label.txt -w weights/reservoir/reservoir3 -s 3 -d Reservoir -l 1
```

### Reservoir 4
```bash
python train.py -p data/Reservoir/Reservoir_4/ -x Reservoir_Data.txt -y Reservoir_Label.txt -w weights/reservoir/reservoir4 -s 4 -d Reservoir -l 1
```

### HVAC 3
```bash
python train.py -p data/HVAC/ROOM_3/ -x HVAC_Data.txt -y HVAC_Label.txt -w weights/hvac/hvac3 -s 3 -d HVAC -l 1

```

### HVAC 6
```bash
python train.py -p data/HVAC/ROOM_6/ -x HVAC_Data.txt -y HVAC_Label.txt -w weights/hvac/hvac6 -s 6 -d HVAC -l 1
```

### Navigation 8x8
```bash
python train.py -p data/Navigation/8x8/ -x Navigation_Data.txt -y Navigation_Label.txt -w weights/nav/8x8 -s 2 -d Navigation -l 2
```

### Navigation 10x10
```bash
python train.py -p data/Navigation/10x10/ -x Navigation_Data.txt -y Navigation_Label.txt -w weights/nav/10x10 -s 2 -d Navigation -l 2
```

## Tensorflow Planning
Planning on trained domain should connected to the real domain similator for evaluation purpose,
Since the learned transition function is not the one in real world. Directly using this planner 
could end up meeting different state in the real world.

Note: 
1. The initial state should be provided by the real world state.
2. The last action given by this planner is not counted in reward, so may be arbitrary. 
3. Change code for the action constraints, not given as parameters currently.

If you only want to check if the planner works in general. Please run following commands with psudo initial state.



### Navigation 8x8
```bash
python plan.py -w weights/nav/8x8 -d Navigation -i Navigation8 -s 2 -a 2 --get_state temp/test/nav/8x8/state --constraint -1 1
```

### Navigation 10x10w
```bash
python plan.py -w weights/nav/10x10 -d Navigation -i Navigation10 -s 2 -a 2 --get_state temp/test/nav/10x10/state --constraint -1 1
```


### HVAC 3
```bash
python plan.py -w weights/hvac/hvac3 -d HVAC -i HVAC3 -s 3 -a 3 --get_state temp/test/hvac/hvac3/state -l 1 --constraint 0 10
```

### HVAC 6
```bash
python plan.py -w weights/hvac/hvac6 -d HVAC -i HVAC6 -s 6 -a 6 --get_state temp/test/hvac/hvac6/state -l 1 --constraint 0 10
```

### Reservoir 3
```bash
python plan.py -w weights/reservoir/reservoir3 -d Reservoir -i Reservoir3 -s 3 -a 3 --get_state temp/test/reservoir/reservoir3/state -l 1
```

### Reservoir 4
```bash
python plan.py -w weights/reservoir/reservoir4 -d Reservoir -i Reservoir4 -s 4 -a 4 --get_state temp/test/reservoir/reservoir4/state -l 1
```