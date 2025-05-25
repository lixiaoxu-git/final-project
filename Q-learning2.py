import numpy as np
import random

#设置迷宫的尺寸
width, height =5,5
def generate_maze(width, height):
    # 初始化迷宫（0表示通路，-1表示墙壁，初始时全是墙壁）
    maze = [[-1 for _ in range(width)] for _ in range(height)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 用DFS方法生成迷宫
    def dfs(x, y):
        maze[y][x] = 0  # 标记为通路
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = x + 2*dx, y + 2*dy
            # 检查新的状态点是否在边界内
            if 0 <= new_x < width and 0 <= new_y < height:
                if maze[new_y][new_x] == -1:  # 如果是未访问的墙壁
                    maze[y+dy][x+dx] = 0      # 打通当前墙
                    dfs(new_x, new_y)         # 递归访问

    # 随机一个起点，并从这个点开始生成迷宫
    start_x, start_y = random.randint(0,width-1),random.randint(0,height-1)
    dfs(start_x, start_y)
    
    # 设置目标点
    while True:
        target_x, target_y = random.randint(0, width-1), random.randint(0, height-1)
        if maze[target_y][target_x] == 0 and (target_x, target_y) != (start_x, start_y):  #确保目标点与起点不重合
            maze[target_y][target_x] = 1
            return maze, (target_x, target_y) ,(start_x,start_y)

maze, (target_x,target_y) ,(start_x,start_y)= generate_maze(width,height)
maze_array = np.array(maze)
maze_array[start_y][start_x] = 2
print(maze_array)

# 创建一个智能体类
class Agent():
    def __init__(self,position):
        self.x = position[0]
        self.y = position[1]
        #定义一个智能体移动的函数
    def move(self,action,maze_array):
        new_x,new_y = self.x + action[0], self.y + action[1]
        #确保智能体不走出迷宫
        if new_y>maze_array.shape[0]-1 or new_x >maze_array.shape[1]-1 or new_x <0 or new_y<0:
            new_x ,new_y = self.x,self.y
        return new_x ,new_y

#初始化
Q_array = np.zeros((width,height,4))
actions = [(0,1),(0,-1),(1,0),(-1,0)]
reward = np.zeros((width,height))
episodes = 1000
learning_rate = 0.05
discount_factor = 0.95
epsilon = 0.1
rows = maze_array.shape[0]
columns = maze_array.shape[1]
#设置即时奖励(移动一步：-1，到达目标点：100，撞墙：-10)
for y in range(rows):
    for x in range(columns):
        if maze_array[y][x] == 0:
            reward[y][x] = -1
        elif maze_array[y][x] == 1:
            reward[y][x] = 100
        elif maze_array[y][x] == -1:
            reward[y][x] = -10

#创建智能体
position  = (start_x,start_y)
agent = Agent(position)

#规定每一个episode只有10000步，防止死循环
step = [10000]*episodes
#创建一个集合记录被访问的位置
for episode in range(episodes):
    agent.x ,agent.y = start_x , start_y
    while step[episode]>0:
        x , y = agent.x, agent.y
        #epsilon-greedy策略（具有一定的探索性）
        if random.random() <epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(Q_array[y, x, :])]
        new_x,new_y = agent.move(action,maze_array)
        current_reward = reward[new_y,new_x]
        
        #更新Q值
        if maze_array[new_y,new_x] == 0:
                Q_array[y,x,actions.index(action)] = Q_array[y,x,actions.index(action)] + learning_rate*(current_reward
                        + discount_factor*max(Q_array[new_y,new_x,:])-Q_array[y,x,actions.index(action)])
                agent.x , agent.y = new_x,new_y
        elif maze_array[new_y,new_x] == -1:
            Q_array[y,x,actions.index(action)] = Q_array[y,x,actions.index(action)] + learning_rate*(current_reward
                        + discount_factor*max(Q_array[y,x,:])-Q_array[y,x,actions.index(action)])
        elif maze_array[new_y,new_x] == 1:
            Q_array[y,x,actions.index(action)] = learning_rate*(current_reward-Q_array[y,x,actions.index(action)])
            break
        step[episode]-=1
if step == [0]*episodes:
    print("无法到达目标位置！")
else:
    print("训练后的Q表:")
    print(Q_array)
    #记录被访问的位置
    visited = set()
    best_path = []
    agent.x, agent.y = start_x ,start_y
    while True:
        x, y = agent.x, agent.y
        #使用greedy策略，寻找最优路径
        current_action = actions[np.argmax(Q_array[y, x, :])]
        best_path.append(current_action)
        
        new_x, new_y = agent.move(current_action,maze_array)
        if (new_x,new_y) in visited:
            print("可能陷入循环")
            break
        visited.add((new_x,new_y))
        if maze_array[new_y, new_x] == 1:
            print("找到目标!")
            break
        
        agent.x, agent.y = new_x, new_y
       
    print("最优路径动作序列:", best_path)