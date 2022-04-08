'''写个代码测试一下学习率随着epoch怎么变的'''
import math
from functools import partial


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10): #warm up学习率预热
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters: #最初阶段
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter: #最后阶段
            lr = min_lr
        else: #中间阶段
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos": #余弦学习率下降
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3) #3
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6) #1.6×1e-3
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15) #15
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)#还需要传入一个iters
    else: #step学习率下降
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size) #还需要传入一个iters

    return func #只需要传入当前epoch数

def set_optimizer_lr(lr_scheduler_func, epoch): # 设置当前这一轮训练的优化器的学习率。传入一个优化器、lr_scheduler_func、当前epoch数
    '''lr_scheduler_func其实就是用get_lr_scheduler在传入学习率下降方式、最小学习率等参数之后得到的，只缺一个epoch参数，出入进去就得到学习率'''
    lr = lr_scheduler_func(epoch)
    return lr

lr_schelur = get_lr_scheduler('step', 1e-3, 1e-4, 100)



list_lr = []
list_epoch = []
for i in range(100):
    lr_get = set_optimizer_lr(lr_schelur,i+1)
    list_lr.append(lr_get)
    list_epoch.append(i+1)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_dict = {}
    for i, j in zip(list_epoch, list_lr):
        data_dict[i] = j

    plt.title("learning rate")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    x = [i for i in data_dict.keys()]
    y = [i for i in data_dict.values()]
    plt.plot(x, y, label="lr")
    plt.legend()
    plt.show()
