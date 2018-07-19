hvac6_instance = {
    'adj_outside': [0, 2, 3, 5],
    'adj_hall': [0, 1, 2, 3, 4, 5],
    'adj': [[0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5]],
    'rooms': [0, 1, 2, 3, 4, 5],
}


hvac3_instance = {
    'adj_outside': [0, 1],
    'adj_hall': [0, 2],
    'adj': [[0, 1], [0, 2], [1, 2]],
    'rooms': [0, 1, 2],
}

reservoir3_instance = {
    "max_cap"          : [100, 200, 400],
    "high_bound"       : [80, 180, 380],
    "low_bound"        : [20, 30, 40],
    "rain"             : [5, 10, 20],
    "downstream"       : [[1, 2], [2, 3]],
    "downtosea"        : [3],
    "biggestmaxcap"    : 1000,
    "reservoirs"       : [1, 2, 3],
    "init_state"       : [75, 50, 50]
}

reservoir4_instance = {
    "max_cap"          : [100, 200, 400, 500],
    "high_bound"       : [80, 180, 380, 480],
    "low_bound"        : [20, 30, 40, 60],
    "rain"             : [5, 10, 20, 30],
    "downstream"       : [[1, 2], [2, 3], [3, 4]],
    "downtosea"        : [4],
    "biggestmaxcap"    : 1000,
    "reservoirs"       : [1, 2, 3, 4],
    "init_state"       : [75, 50, 50]
}

navi10_instance = {
    "dims": 2,
    "min_maze_bound": -5.0,
    "max_maze_bound": 5.0,
    "min_act_bound": -1.0,
    "max_act_bound": 1,
    "goal": 3.0,
    "centre": 0
}

navi8_instance = {
    "dims": 2,
    "min_maze_bound": -4.0,
    "max_maze_bound": 4.0,
    "min_act_bound": -1.0,
    "max_act_bound": 1,
    "goal": 3.0,
    "centre": 0
}
