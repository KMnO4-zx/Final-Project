import random
import numpy as np

class Environment:
    def __init__(self):
        # 初始化各类参数
        self.counts = 0
        self.time_windows = []
        self.total_time = 0.0
        self.max_time = 1200
        self.total_storage = 0.0
        self.max_storage = 175
        self.each_storage = 5
        self.total_task = 100
        self.transfer_time = 5.0
        self.total_profit = 0
        self.atasks = 0
        self.task = []
        
        # 初始化任务的开始和结束时间
        self.es = 10
        self.ee = 15
        
        # 生成随机任务并初始化
        self._generate_tasks()

        # 初始状态
        self.state = None
        self.done = False

    def _generate_tasks(self):
        """生成任务的随机到达时间、持续时间、存储需求和利润"""
        arrival_times = sorted(random.randint(0, self.max_time) for _ in range(self.total_task))
        for arrival in arrival_times:
            duration = random.randint(self.es, self.ee)
            profit = random.randint(1, 10)
            task = [arrival, duration, self.each_storage, profit, arrival, arrival + duration]
            self.task.append(task)

    def update_env(self, action):
        """更新环境状态，根据输入的action（0为接受任务，1为拒绝任务）"""
        self.atasks = len(self.time_windows)
        self.done = self.total_storage > self.max_storage or self.counts >= self.total_task

        if action == 1:  # 拒绝任务
            return self._next_state(reject=True)

        # 设置任务的开始和结束时间
        if self.atasks == 0:
            self._assign_task_times(self.counts, start_now=True)
        else:
            last_task_end = self.task[self.time_windows[-1]][5]
            remaining_time = self.max_time - last_task_end
            required_time = 2 * self.transfer_time + self.task[self.counts][1]

            if remaining_time > required_time and last_task_end + 2 * self.transfer_time <= self.task[self.counts][0]:
                return self._next_state()
            elif self.task[self.counts][5] > self.max_time:  # 任务结束时间超出最大时间
                return self._next_state(reject=True)  # 拒绝任务

        # 更新环境状态
        self.total_storage += self.task[self.counts][2]
        if self.total_storage > self.max_storage:
            return np.array([0, 0, 0, 0]), self.total_profit, self.done

        self.total_time += self.task[self.counts][1]
        self.time_windows.append(self.counts)
        self.total_profit += self.task[self.counts][3]
        return self._next_state()

    def _assign_task_times(self, task_idx, start_now=False):
        """设置任务的开始和结束时间"""
        if start_now:
            self.task[task_idx][4] = self.task[task_idx][0]
            self.task[task_idx][5] = self.task[task_idx][4] + self.task[task_idx][1]

    def _next_state(self, reject=False):
        """返回下一个状态"""
        if reject:
            self.counts += 1
        
        # 将状态标准化为 [0,1]
        normalized_arrival_time = self.task[self.counts][0] / self.max_time
        normalized_profit = self.task[self.counts][3] / 10
        normalized_storage = self.total_storage / self.max_storage
        normalized_time = (self.max_time - self.total_time) / self.max_time
        
        # 下一个状态
        next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
        return np.array(next_state), self.total_profit, self.done

    def reset(self):
        """重置环境状态"""
        self.time_windows.clear()
        self.counts = 0
        self.atasks = 0
        self.total_storage = 0
        self.total_profit = 0
        self.done = False
        self.total_time = 0.0 
        self._generate_tasks()  # 重新生成任务
        self.state = None

    def observe(self):
        """返回当前状态"""
        if self.counts < self.total_task:
            normalized_arrival_time = self.task[self.counts][0] / self.max_time
            normalized_profit = self.task[self.counts][3] / 10
            normalized_storage = self.total_storage / self.max_storage
            normalized_time = (self.max_time - self.total_time) / self.max_time
            current_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
            return np.array(current_state)
        else:
            return np.array([0, 0, 0, 0])  # 如果没有任务则返回零状态
        
if __name__ == "__main__":
    env = Environment()
    print("Initial State:", env.observe())

    # Simulate a sequence of actions (0 for accepting, 1 for rejecting)
    actions = [0, 1, 0, 0, 1, 0]  # Example action sequence
    for i, action in enumerate(actions):
        next_state, total_profit, done = env.update_env(action)
        print(f"Step {i+1}: Action: {action}, Next State: {next_state}, Total Profit: {total_profit}, Done: {done}")
        if done:
            break  # Stop if done

    # Reset the environment
    env.reset()
    print("After Reset State:", env.observe())

    # Check the state after reset
    next_state_after_reset = env.observe()
    print("Next State After Reset:", next_state_after_reset)
