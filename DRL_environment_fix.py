import numpy as np
import random

class Environment(object):
    # 初始化环境的所有参数和变量
    def __init__(self, 
                    counts: int = 0, # 任务计数器
                    max_time: int = 1200, # 卫星的最大时间跨度（可用时间）
                    free_time: int = 1200, # 初始空闲时间
                    max_storage: int = 175, # 卫星的最大存储容量
                    each_storage: int = 5, # 每个任务的存储消耗
                    total_task: int = 100, # 总任务数
                    transfer_time: float = 5.0, # 任务之间的转移时间
                    es: int = 10, # 观测任务的开始时间范围
                    ee: int = 15, # 观测任务的结束时间范围
                ):   
        self.counts = counts
        self.max_time = max_time
        self.free_time = free_time
        self.max_storage = max_storage
        self.each_storage = each_storage
        self.total_task = total_task
        self.transfer_time = transfer_time
        self.es = es
        self.ee = ee
        # 用于存储已接受任务的时间窗口列表
        self.time_windows = []
        # 总使用时间初始化为0
        self.total_time = 0.0
        # 当前已使用的总存储量
        self.total_storage = 0.
        # 初始总收益
        self.total_profit = 0
        self.et = 0
        # 已接受任务的数量
        self.atasks = 0
        # 任务列表
        self.task = []
        # 任务到达时间列表
        self.arrival = []
        # 初始化任务到达时间，随机生成并排序
        for i in range(self.total_task):
            a = random.randint(0, self.max_time)
            self.arrival.append(a)
        self.arrival.sort()
        # 初始化任务
        for i in range(self.total_task):
            self.task.append([])
        self.each_storage = 5
        # 给每个任务赋予属性：到达时间、持续时间、存储需求、收益、开始时间、结束时间
        for i in range(self.total_task):
            self.task[i].append(self.arrival[i])
            self.et = random.randint(self.es, self.ee)
            self.task[i].append(self.et)
            self.task[i].append(self.each_storage)
            self.profit = random.randint(1, 10)
            self.task[i].append(self.profit)
            self.task[i].append(self.arrival[i])
            self.task[i].append(self.arrival[i] + self.et)
        # 初始状态为空
        self.state = None

    
    