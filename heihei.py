import numpy as np
import time


def readInput(inputFile):  # read input.txt
    lines = []
    with open(inputFile, 'r') as f:
        for line in f:
            line = line.strip('\n')
            lines.append(line)

    Grid_size = lines[0]
    Wall_cells_num = int(lines[1])
    Wall_cells_pos = lines[2:2 + Wall_cells_num]
    Term_states_num = int(lines[2 + Wall_cells_num])
    Term_states_pos = lines[3 + Wall_cells_num:3 + Wall_cells_num + Term_states_num]
    Tran_model = lines[3 + Wall_cells_num + Term_states_num]
    Rewards = lines[4 + Wall_cells_num + Term_states_num]
    Discount = np.float_(lines[5 + Wall_cells_num + Term_states_num])
    return [Grid_size, Wall_cells_num, Wall_cells_pos, Term_states_num, Term_states_pos, Tran_model, Rewards, Discount]


def dataProcessing(Grid_sz, Wall_num, Wall_pos, Term_num, Term_pos, Tran_m, Rw):
    Grid_sz = Grid_sz.split(",")
    N_row, N_col = int(Grid_sz[0]), int(Grid_sz[1])
    Maze = np.zeros((N_row, N_col))
    Uti = np.zeros((N_row, N_col))
    Wall = []
    for ele in Wall_pos:
        ele = ele.split(" ")
        for ele2 in ele:
            ele2 = ele2.split(",")
            pnt = [int(ele2[0]), int(ele2[1])]
            Maze[N_row - int(ele2[0]), int(ele2[1]) - 1] = 1  # 1 wall, 0 space, 2 terminal
            Wall.append(pnt)
    assert len(Wall) == Wall_num
    # print(Maze.shape)
    Term = []
    for ele in Term_pos:
        ele = ele.split(" ")
        for ele2 in ele:
            ele2 = ele2.split(",")
            pnt = [int(ele2[0]), int(ele2[1])]
            Maze[N_row - int(ele2[0]), int(ele2[1]) - 1] = 2
            # print(ele2)
            Uti[N_row - int(ele2[0]), int(ele2[1]) - 1] = np.float_(ele2[2])
            Term.append(pnt)
    assert len(Term) == Term_num
    Tran_m = Tran_m.split(",")
    P_walk, P_run = np.float_(Tran_m[0]), np.float_(Tran_m[1])
    P_walk_f = (1 - P_walk) / 2
    P_run_f = (1 - P_run) / 2
    Rw = Rw.split(",")
    R_walk, R_run = np.float_(Rw[0]), np.float_(Rw[1])
    # print(Maze)
    # print(Uti)
    return [N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti]


def validPos(Cur_row, Cur_col, action, Maze, N_row, N_col, Uti):
    res = []
    if action == 0:  # Walk UP
        if Cur_row == 0 or Maze[Cur_row - 1, Cur_col] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row - 1, Cur_col]
        if Cur_col == 0 or Maze[Cur_row, Cur_col - 1] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row, Cur_col - 1]
        if Cur_col == N_col - 1 or Maze[Cur_row, Cur_col + 1] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row, Cur_col + 1]
    elif action == 1:  # Walk DOWN
        if Cur_row == N_row - 1 or Maze[Cur_row + 1, Cur_col] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row + 1, Cur_col]
        if Cur_col == N_col - 1 or Maze[Cur_row, Cur_col + 1] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row, Cur_col + 1]
        if Cur_col == 0 or Maze[Cur_row, Cur_col - 1] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row, Cur_col - 1]
    elif action == 2:  # Walk LEFT
        if Cur_col == 0 or Maze[Cur_row, Cur_col - 1] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row, Cur_col - 1]
        if Cur_row == N_row - 1 or Maze[Cur_row + 1, Cur_col] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row + 1, Cur_col]
        if Cur_row == 0 or Maze[Cur_row - 1, Cur_col] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row - 1, Cur_col]
    elif action == 3:  # Walk RIGHT
        if Cur_col == N_col - 1 or Maze[Cur_row, Cur_col + 1] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row, Cur_col + 1]
        if Cur_row == 0 or Maze[Cur_row - 1, Cur_col] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row - 1, Cur_col]
        if Cur_row == N_row - 1 or Maze[Cur_row + 1, Cur_col] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row + 1, Cur_col]
    elif action == 4:  # Run UP
        if Cur_row <= 1 or Maze[Cur_row - 1, Cur_col] == 1 or Maze[Cur_row - 2, Cur_col] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row - 2, Cur_col]
        if Cur_col <= 1 or Maze[Cur_row, Cur_col - 1] == 1 or Maze[Cur_row, Cur_col - 2] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row, Cur_col - 2]
        if Cur_col >= N_col - 2 or Maze[Cur_row, Cur_col + 1] == 1 or Maze[Cur_row, Cur_col + 2] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row, Cur_col + 2]
    elif action == 5:  # Run DOWN
        if Cur_row >= N_row - 2 or Maze[Cur_row + 1, Cur_col] == 1 or Maze[Cur_row + 2, Cur_col] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row + 2, Cur_col]
        if Cur_col >= N_col - 2 or Maze[Cur_row, Cur_col + 1] == 1 or Maze[Cur_row, Cur_col + 2] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row, Cur_col + 2]
        if Cur_col <= 1 or Maze[Cur_row, Cur_col - 1] == 1 or Maze[Cur_row, Cur_col - 2] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row, Cur_col - 2]
    elif action == 6:  # Run LEFT
        if Cur_col <= 1 or Maze[Cur_row, Cur_col - 1] == 1 or Maze[Cur_row, Cur_col - 2] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row, Cur_col - 2]
        if Cur_row >= N_row - 2 or Maze[Cur_row + 1, Cur_col] == 1 or Maze[Cur_row + 2, Cur_col] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row + 2, Cur_col]
        if Cur_row <= 1 or Maze[Cur_row - 1, Cur_col] == 1 or Maze[Cur_row - 2, Cur_col] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row - 2, Cur_col]
    elif action == 7:  # Run RIGHT
        if Cur_col >= N_col - 2 or Maze[Cur_row, Cur_col + 1] == 1 or Maze[Cur_row, Cur_col + 2] == 1:  # main direction
            pos0 = [Cur_row, Cur_col]
        else:
            pos0 = [Cur_row, Cur_col + 2]
        if Cur_row <= 1 or Maze[Cur_row - 1, Cur_col] == 1 or Maze[Cur_row - 2, Cur_col] == 1:  # +90
            pos1 = [Cur_row, Cur_col]
        else:
            pos1 = [Cur_row - 2, Cur_col]
        if Cur_row >= N_row - 2 or Maze[Cur_row + 1, Cur_col] == 1 or Maze[Cur_row + 2, Cur_col] == 1:  # -90
            pos2 = [Cur_row, Cur_col]
        else:
            pos2 = [Cur_row + 2, Cur_col]
    res.append(Uti[pos0[0], pos0[1]])
    res.append(Uti[pos1[0], pos1[1]])
    res.append(Uti[pos2[0], pos2[1]])
    return res  # The utility of 3 directions


def training(N_row, N_col, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti, Disc, Max_iter=1e2, error=1e-3):
    Uti_old = Uti.copy()
    # idx = np.where(Uti_old==0)
    # Uti_old[idx] = R_walk
    # print(Uti_old)
    Pol = np.zeros((N_row, N_col))
    Uti_new = Uti_old.copy()
    for i in range(Max_iter):
        for x in range(N_row):
            for y in range(N_col):
                if Maze[x, y] == 1:
                    # print(x, y)
                    Pol[x, y] = 8
                    continue
                if Maze[x, y] == 2:
                    # print(x, y)
                    Pol[x, y] = 9
                    continue
                max_act_uti = -100
                best_act = 0
                for act in range(8):
                    pos_list = validPos(x, y, act, Maze, N_row, N_col, Uti_old)  # 0410
                    # if x==2 and y==3:
                    #     print(pos_list)
                    cur_act_uti = 0
                    # max_act_uti = -100
                    # best_act = 0
                    if act < 4:  # Walk
                        P = P_walk
                        P_f = P_walk_f
                        R = R_walk
                    else:  # Run
                        P = P_run
                        P_f = P_run_f
                        R = R_run
                    for j in range(len(pos_list)):
                        if j == 0:
                            cur_act_uti = pos_list[j] * P
                        else:
                            cur_act_uti = cur_act_uti + pos_list[j] * P_f
                    cur_act_uti = cur_act_uti * Disc + R
                    # if i<=4 and x == 1 and y==58:
                    #     print(act)
                    #     print(cur_act_uti)
                    if cur_act_uti == max_act_uti:  # preference of the move
                        if act <= best_act:
                            best_act = act
                    if cur_act_uti - max_act_uti > 1e-8:
                        max_act_uti = cur_act_uti
                        best_act = act
                    # if x == 1 and y == 2 and i == 1:
                    #     print(pos_list)
                    #     print(act)
                    #     print(cur_act_uti)
                    #     print(max_act_uti)

                Uti_new[x, y] = max_act_uti
                Pol[x, y] = best_act

        # print(Pol[1][58])
        if i <= 10:
            # print(Uti_new[0:5, 55:60])
            print(Uti_new)
            # print(Pol)
        # print(i)
        # print(Pol[0, :])
        # print(Uti_old)
        # print(Uti_new[0, 4])
        # print(x, y)
        diff = np.sum(np.abs(Uti_new - Uti_old))
        if np.max(diff) < error:
            break
        # if np.sum(Pol_old - Pol_new) == 0:
        #     break
        Uti_old = Uti_new.copy()
        # Pol_old = Pol_new.copy()
    num_iter = i + 1
    # print(Uti_new[0][4])
    return Uti_new, Pol, num_iter


start = time.clock()
[Grid_sz, Wall_num, Wall_pos, Term_num, Term_pos, Tran_m, Rw, Disc] = readInput("input1.txt")
[N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti] = dataProcessing(Grid_sz, Wall_num, Wall_pos, Term_num, Term_pos, Tran_m, Rw)
Uti_final, Pol_final, N_final = training(N_row, N_col, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti, Disc, 500, 1e-4)
# print(N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze[400, 99], Uti[400, 99])
print(N_final)
# print(Uti_final)
# print(Pol_final)
# printMove(Pol_final)

with open('output222.txt', 'w') as f:
    for i in range(N_row):
        for j in range(N_col):
            if Pol_final[i][j] == 0:
                str = 'Walk Up'
            elif Pol_final[i][j] == 1:
                str = 'Walk Down'
            elif Pol_final[i][j] == 2:
                str = 'Walk Left'
            elif Pol_final[i][j] == 3:
                str = 'Walk Right'
            elif Pol_final[i][j] == 4:
                str = 'Run Up'
            elif Pol_final[i][j] == 5:
                str = 'Run Down'
            elif Pol_final[i][j] == 6:
                str = 'Run Left'
            elif Pol_final[i][j] == 7:
                str = 'Run Right'
            elif Pol_final[i][j] == 8:
                str = 'None'
            elif Pol_final[i][j] == 9:
                str = 'Exit'
            if j == N_col - 1:
                str += "\n"
                # if i != N_row - 1:
                #     str += "\n"
            else:
                str += ","
            f.write(str)
elapsed = (time.clock() - start)
print("Time used:", elapsed)

