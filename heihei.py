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
        # if i is 0:
        print(Uti_new[0:10, -10:-1])
        #     print(Uti_new[0][4])
        #     print(Pol[0:8, 52:60])
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
[Grid_sz, Wall_num, Wall_pos, Term_num, Term_pos, Tran_m, Rw, Disc] = readInput("input1_no_wall.txt")
[N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti] = dataProcessing(Grid_sz, Wall_num, Wall_pos, Term_num, Term_pos, Tran_m, Rw)
Uti_final, Pol_final, N_final = training(N_row, N_col, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze, Uti, Disc, 500, 1e-4)
# print(N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run, P_walk_f, P_run_f, Maze[400, 99], Uti[400, 99])
print(N_final)
# print(Uti_final)
# print(Pol_final)
# printMove(Pol_final)

with open('output2_no_wall.txt', 'w') as f:
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

#     f.write(Grid_sz)
#     f.write('\n')
#     f.write(str(Wall_num))
#     f.write('\n')
#     f.write(W)
#     f.write('\n')
#     f.write(str(Term_num))
#     f.write('\n')
#     f.write(T)
#     f.write('\n')
#     f.write(Tran_m)
#     f.write('\n')
#     f.write(Rw)
#     f.write('\n')
#     f.write(Disc)
#     f.close()
# def printMove(Pol):
#     for ele in Pol:
#         for ele2 in ele:
#             if ele2 == 0:
#                 print('Walk Up')
#             elif ele2 == 1:
#                 print('Walk Down')
#             elif ele2 == 2:
#                 print('Walk Left')
#             elif ele2 == 3:
#                 print('Walk Right')
#             elif ele2 == 4:
#                 print('None')
#             elif ele2 == 5:
#                 print('Exit')
#         print("\n")
# W = " ".join(Wall_pos)
# T = " ".join(Term_pos)
# print([N_row, N_col, Wall, Term, P_walk, P_run, R_walk, R_run])

# print(Grid_sz)
# print(Disc)

# import numpy as np
# import time
#
#
# class MDP:
#     def policy_evaluation(self, action):
#         new_map = None
#         if action == self.walk_up:
#             new_map = np.delete(np.concatenate((self.value[0:1, ...], self.value), axis=0), -1, axis=0)
#             # self.value = np.multiply(self.p[:,:,self.up, action], new_map) +
#         elif action == self.walk_down:
#             new_map = np.delete(np.concatenate((self.value, self.value[-1:, ...]), axis=0), 0, axis=0)
#         elif action == self.walk_left:
#             new_map = np.delete(np.concatenate((self.value[..., 0:1], self.value), axis=1), -1, axis=1)
#         elif action == self.walk_right:
#             new_map = np.delete(np.concatenate((self.value, self.value[..., -1:]), axis=1), 0, axis=1)
#         elif action == self.run_up:
#             new_map = np.delete(np.delete(np.concatenate((self.value[0:2:, ...], self.value), axis=0), -1, axis=0), -1, axis=0)
#         elif action == self.run_down:
#             new_map = np.delete(np.concatenate((self.value, self.value[-2:, ...]), axis=0), 1, axis=0)
#             new_map = np.delete(new_map, 1, axis=0)
#         elif action == self.run_left:
#             new_map = np.delete(np.concatenate((self.value[..., 0:2], self.value), axis=1), -1, axis=1)
#             new_map = np.delete(new_map, -1, axis=1)
#         elif action == self.run_right:
#             new_map = np.delete(np.concatenate((self.value, self.value[..., -2:]), axis=1), -1, axis=1)
#             new_map = np.delete(new_map, -1, axis=1)
#         return np.multiply(self.p[:, :, self.up, action].flatten().reshape(self.length, self.width), new_map) + \
#                np.multiply(self.p[:, :, self.down, action].flatten().reshape(self.length, self.width), new_map) \
#                + np.multiply(self.p[:, :, self.left, action].flatten().reshape(self.length, self.width), new_map) + \
#                np.multiply(self.p[:, :, self.right, action].flatten().reshape(self.length, self.width), new_map)
#
#     def value_iteration(self):
#         u_p = 0
#         delta = 0
#         print(time.time())
#         while delta < self.e * (1 - self.discount) / self.discount:
#             print(time.time())
#             u = u_p
#             delta = 0
#             max = np.full((self.length, self.width), -10000)
#             for action in range(8):
#                 if action < 4:
#                     u_p = np.maximum(max, self.policy_evaluation(action)) + self.reward_walk
#                 else:
#                     u_p = np.maximum(max, self.policy_evaluation(action)) + self.reward_walk
#                 if np.min(np.abs(u - u_p)) > delta:
#                     delta = np.min(np.abs(u - u_p))
#
#     def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
#         self.discount = discount
#         self.reward_run = reward_run
#         self.reward_walk = reward_walk
#         self.length = length
#         self.width = width
#         self.e = e
#         self.up = 0,
#         self.left = 1,
#         self.down = 2
#         self.right = 3
#
#         self.walk_up = 0
#         self.walk_left = 1
#         self.walk_down = 2
#         self.walk_right = 3
#
#         self.run_up = 4
#         self.run_left = 5
#         self.run_down = 6
#         self.run_right = 7
#
#         self.p = np.zeros((length, width, 4, 8))  # four directions and 8 actions
#
#         self.p[:, :, self.up, self.walk_up] = p_walk
#         self.p[:, :, self.left, self.walk_up] = .5 * (1 - p_walk)
#         self.p[:, :, self.right, self.walk_up] = .5 * (1 - p_walk)
#
#         self.p[:, :, self.right, self.walk_right] = p_walk
#         self.p[:, :, self.up, self.walk_left] = .5 * (1 - p_walk)
#         self.p[:, :, self.down, self.walk_left] = .5 * (1 - p_walk)
#
#         self.p[:, :, self.down, self.walk_down] = p_walk
#         self.p[:, :, self.left, self.walk_down] = .5 * (1 - p_walk)
#         self.p[:, :, self.right, self.walk_down] = .5 * (1 - p_walk)
#
#         self.p[:, :, self.left, self.walk_left] = p_walk
#         self.p[:, :, self.up, self.walk_down] = .5 * (1 - p_walk)
#         self.p[:, :, self.down, self.walk_down] = .5 * (1 - p_walk)
#
#         self.p[:, :, self.up, self.run_up] = p_run
#         self.p[:, :, self.left, self.run_up] = .5 * (1 - p_run)
#         self.p[:, :, self.right, self.run_up] = .5 * (1 - p_run)
#
#         self.p[:, :, self.right, self.run_right] = p_run
#         self.p[:, :, self.up, self.run_right] = .5 * (1 - p_run)
#         self.p[:, :, self.down, self.run_right] = .5 * (1 - p_run)
#
#         self.p[:, :, self.down, self.run_down] = p_run
#         self.p[:, :, self.left, self.run_down] = .5 * (1 - p_run)
#         self.p[:, :, self.right, self.run_down] = .5 * (1 - p_run)
#
#         self.p[:, :, self.left, self.run_left] = p_run
#         self.p[:, :, self.up, self.run_left] = .5 * (1 - p_run)
#         self.p[:, :, self.down, self.run_left] = .5 * (1 - p_run)
#
#         self.policy = np.zeros((length, width))
#         self.value = np.zeros((length, width))
#         for exit in exit_list:
#             location = exit[0]
#             utility = exit[1]
#             self.value[location[0], location[1]] = utility
#
#
# def main():
#     # def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
#     mdp = MDP(length=500, width=600, p_walk=1, p_run=1, reward_run=0, reward_walk=0, discount=0.7, exit_list=[((100, 100), 10)])
#     mdp.value_iteration()
#
#
# if __name__ == "__main__":
#     main()
