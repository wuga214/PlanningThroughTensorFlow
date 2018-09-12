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