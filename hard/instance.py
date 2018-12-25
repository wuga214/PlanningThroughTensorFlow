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

hvac60_instance = {
'adj_hall' : [101,102,103,106,107,109,110,
               201,202,203,206,207,209,210,
               301,302,303,306,307,309,310,
               401,402,403,406,407,409,410,
               501,502,503,506,507,509,510],
'adj_outside' : [101,102,103,104,105,106,108,110,111,112,
              201,202,203,204,205,206,208,210,211,212,
              301,302,303,304,305,306,308,310,311,312,
              401,402,403,404,405,406,408,410,411,412,
              501,502,503,504,505,506,508,510,511,512],
'adj' : [[101,102],[102,103],[103,104],[104,105],[106,107],[107,108],[107,109],[108,109],[110,111],[111,112],
       [201,202],[202,203],[203,204],[204,205],[206,207],[207,208],[207,209],[208,209],[210,211],[211,212],
       [301,302],[302,303],[303,304],[304,305],[306,307],[307,308],[307,309],[308,309],[310,311],[311,312],
       [401,402],[402,403],[403,404],[404,405],[406,407],[407,408],[407,409],[408,409],[410,411],[411,412],
       [501,502],[502,503],[503,504],[504,505],[506,507],[507,508],[507,509],[508,509],[510,511],[511,512],
       [101,201],[102,202],[103,203],[104,204],[105,205],[106,206],[107,207],[108,208],[109,209],[110,210],
       [111,211],[112,212],[201,301],[202,302],[203,303],[204,304],[205,305],[206,306],[207,307],[208,308],
       [209,309],[210,310],[211,311],[212,312],[301,401],[302,402],[303,403],[304,404],[305,405],[306,406],
       [307,407],[308,408],[309,409],[310,410],[311,411],[312,412],[401,501],[402,502],[403,503],[404,504],
       [405,505],[406,506],[407,507],[408,508],[409,509],[410,510],[411,511],[412,512]],
'rooms' : list(range(101,113))+list(range(201,213))+list(range(301,313))+list(range(401,413))+list(range(501,513))
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
