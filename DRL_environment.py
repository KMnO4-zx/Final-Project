"""
Description:
Online scheduling of image satellites based on deep reinforcement learning:
    
Environment:
The situation is based on a limited time span, there is an image satellite, which has a limited storage and observe tasks within time windows, 
the satellite’s time windows for the tasks can be got by STK, which is a professional software to get scientific data,task arrives dynamically. 
Each task has its own storage,profit.The reward of the selected task will be added if the it’s selected successfully. 
The aim of the problem is try to get maximum profit by selecting the task reasonably.
    
Actions:
Type: 
Num: for each task arrive dynamiclly, the satellites will accept or not    
Using a number to show the action,for example 0 means accept this task, 1 means doesn't accept this task.       

Reward:
The initial reward is zero, then if a task is accepted,the value of the task’s profit will be added into the total reward.

Starting State:
the start state is a satellite which does't accept any task:  []

Episode Termination:
1.When this total weight is larger than the  maximum  storage of the satellites 
2.Time
3.Task number


Description: 在线调度遥感卫星基于深度强化学习：

Environment:
问题背景是一个有限的时间范围内，有一颗遥感卫星，卫星的存储容量有限，并且需要在指定的时间窗口内观测任务。
卫星任务的时间窗口可通过STK（一种获取科学数据的专业软件）获取，任务会动态到达。每个任务都有其对应的存储需求和收益值。
当一个任务被成功选择时，其收益将被累加到总收益中。问题的目标是通过合理选择任务，以获取最大的总收益。

Actions：
- **类型：** 对于每个动态到达的任务，卫星可以选择接受或不接受。
- **数量：** 用数字表示动作，例如，0表示接受任务，1表示不接受任务。

Reward：
初始奖励为零，如果一个任务被接受，该任务的收益值将被加到总奖励中。

Starting State：
起始状态是一个未接受任何任务的卫星： `[]`

Episode Termination：
1. 当总存储超过卫星的最大存储容量时
2. 时间到达上限
3. 任务数量达上限

"""
import numpy as np
import random
 
class Environment(object):
# introduce and initialize all the parameters and variables of the environment
    def __init__(self):   
    # count
        self.counts = 0
    #use a list to collect tasks
        self.time_windows=[]
    # initial the total used time
        self.total_time=0.0
    # the time span of the satellite
        self.max_time = 1200
    # initial the free time
        self.free_time = 1200
    # the total storage at the present:
        self.total_storage = 0.0
    # the maximum storage 
        self.max_storage = 175
        # self.max_storage = 250 1200_100
    # storage consumption Stori
        self.each_storage =5
    # the total task 
        self.total_task = 100
    # transfer time 
        self.transfer_time = 5.0
    # initial profit
        self.total_profit = 0
        self.et = 0
    # count the total number of acc
        self.atasks=0
    # initial start and end observation time
        self.es=10
        self.ee=15  
        self.task=[]
        self.arrival=[]
        for i in range(self.total_task):
            a= random.randint(0,self.max_time)
            self.arrival.append(a)
        self.arrival.sort()
        for i in range(self.total_task):
            self.task.append([])
        self.each_storage = 5
        for i in range(self.total_task):
            self.task[i].append(self.arrival[i])
            #self.et= random.randint(20, 40)
            self.et= random.randint(self.es, self.ee)
            self.task[i].append(self.et)
            self.task[i].append(self.each_storage)
            self.profit = random.randint(1,10)
            self.task[i].append(self.profit)
            self.task[i].append(self.arrival[i])
            self.task[i].append(self.arrival[i]+self.et)
    #initial the state
        self.state = None
        
    # method which can update the environment after get a new action       
    def update_env(self,action):#action['task_number']
        self.atasks = len(self.time_windows)
        done = bool(
            self.total_storage > self.max_storage
            or self.counts >= self.total_task
        )
        self.done =bool(done)

        #Not accept
        if(action==1):
            self.counts = self.counts+1
            #print(self.counts)
            #normalize the state to [0,1]
            normalized_arrival_time= self.task[self.counts][0]/self.max_time
            normalized_profit= self.task[self.counts][3]/10
            normalized_storage = self.total_storage/self.max_storage
            normalized_time = (self.max_time-self.total_time)/self.max_time
            next_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
            return np.array(next_state), self.total_profit, self.done
        else:
            #set start and end time
            if(self.atasks==0): 
                self.task[self.counts][4]=self.task[self.counts][0]
                self.task[self.counts][5]=self.task[self.counts][4]+self.task[self.counts][1]
               
            else:#check if there is a enough time to asign the current task
                #查看是否还有足够的时间来分配当前任务
                #关于空闲时间的处理
                count2 = self.time_windows[self.atasks-1]
                t = self.max_time - self.task[count2][5]
                q = 2*self.transfer_time+self.task[self.counts][1]

                if(t>q):#比较最后一个任务的结束时间和当前任务的到达时间是否冲突
                    if(self.task[count2][5]+2*self.transfer_time>self.task[self.counts][0]):
                        self.counts = self.counts+1
                        #normalize the state to [0,1]
                        normalized_arrival_time= self.task[self.counts][0]/self.max_time
                        normalized_profit= self.task[self.counts][3]/10
                        normalized_storage = self.total_storage/self.max_storage
                        normalized_time = (self.max_time-self.total_time)/self.max_time
                        next_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
                        return np.array(next_state), self.total_profit, self.done
                else:
                    self.counts = self.counts+1
                    #normalize the state to [0,1]
                    normalized_arrival_time= self.task[self.counts][0]/self.max_time
                    normalized_profit= self.task[self.counts][3]/10
                    normalized_storage = self.total_storage/self.max_storage
                    normalized_time = (self.max_time-self.total_time)/self.max_time
                    next_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
                    return np.array(next_state), self.total_profit, self.done
                if(self.task[self.counts][5]> self.max_time):
                    self.counts = self.counts+1
                    #normalize the state to [0,1]
                    normalized_arrival_time= self.task[self.counts][0]/self.max_time
                    normalized_profit= self.task[self.counts][3]/10
                    normalized_storage = self.total_storage/self.max_storage
                    normalized_time = (self.max_time-self.total_time)/self.max_time
                    next_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
                    return np.array(next_state), self.total_profit, self.done

        # Updating the total used storage
        self.total_storage += self.task[self.counts][2]
        if(self.total_storage>self.max_storage):
            next_state=(0,0,0,0)
            return np.array(next_state), self.total_profit, self.done 
        # Updating the total taken time
        self.total_time +=self.task[self.counts][1]
        
        #Update the reward and add task
        self.time_windows = self.time_windows +[self.counts]
        self.total_profit = self.total_profit +self.task[self.counts][3]
        self.counts = self.counts+1
        
        #normalize the state to [0,1]
        normalized_arrival_time= self.task[self.counts][0]/self.max_time
        normalized_profit= self.task[self.counts][3]/10
        normalized_storage = self.total_storage/self.max_storage
        normalized_time = (self.max_time-self.total_time)/self.max_time
        next_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
        return np.array(next_state), self.total_profit, self.done
    
    # A method used to reset the environment
    def reset(self):
        self.time_windows.clear()
       
        self.counts=0
        self.atasks=0
        
        self.total_storage=0
        self.total_profit=0
        self.done = False
        self.total_time=0.0 
        self.arrival.clear()
        self.task.clear()
        self.et = 0
        for i in range(self.total_task):
            a= random.randint(0,self.max_time)
            self.arrival.append(a)
        self.arrival.sort()
        for i in range(self.total_task):
            self.task.append([])
        self.each_storage = 5
        for i in range(self.total_task):
            self.task[i].append( self.arrival[i])
            #self.et= random.randint(10, 15)
            self.et= random.randint(self.es, self.ee)
            self.task[i].append(self.et)
            self.task[i].append(self.each_storage)
            self.profit = random.randint(1,10)
            self.task[i].append(self.profit)
            self.task[i].append(self.arrival[i])
            self.task[i].append(self.arrival[i]+self.et)
        self.state = None

   #A method used to return the first state
    def observe(self):
        normalized_arrival_time= self.task[0][0]/self.max_time
        normalized_profit= self.task[0][3]/10
        normalized_storage = self.total_storage/self.max_storage
        normalized_time = (self.max_time-self.total_time)/self.max_time
        current_state = (normalized_arrival_time,normalized_profit,normalized_storage,normalized_time)
        return np.array(current_state)
        
    
if __name__ == "__main__":
    # 创建环境实例
    env = Environment()
    # 重置环境
    state = env.reset()
    print("初始状态:", state)
    
    # 模拟几次动作
    for _ in range(10):
        action = random.randint(0, 1)  # 随机选择接受或拒绝任务
        next_state, reward, done = env.update_env(action)
        print(f"选择的动作: {action}, 下一状态: {next_state}, 当前收益: {reward}, 是否结束: {done}")
        if done:
            break
    print("最终收益:", env.total_profit)

    
    