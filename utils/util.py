# util.py

import matplotlib
import numpy as np
import os
import math


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            matplotlib.pyplot.plot(x, np.asarray(numbers[name]))
        matplotlib.pyplot.legend([self.title + '(' + name + ')' for name in names])
        matplotlib.pyplot.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def visualize(feat, target_labels, sensitive_labels, epoch, arg_plt=None):
    if arg_plt is None:
        matplotlib.pyplot.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    c = ['#ff0000', '#009999', '#0000ff', '#00ff00', '#999900',
         '#ffff00', '#ff00ff',  '990000',  '#00ffff','#009900']
    matplotlib.pyplot.clf()
    m = 1 + np.max(target_labels)
    n = 1 + np.max(sensitive_labels)

    # c = get_spaced_colors(m)
    # c = np.asarray(c)
    # c = c/256
    marker = ['o', 's', 'v', 'P', 'H', 'D', '+', 'x', '^', '<', '>', '1', '2', '3', '4', '8', '.', 'p', '*', 'h', 'X',
              'd', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, '$..$','o']

    legend_str = []
    for i in range(m):
        for j in range(n):
            index = (target_labels == i) & (sensitive_labels == j)
            matplotlib.pyplot.plot(feat[index, 0], feat[index, 1], '.', c=c[j], marker=marker[i])
            # legend_str.append(str(i)+str(j))

    legend_str = ['%d' % number for number in np.arange(0, n)]
    # a = np.arange(m)
    # legend_str = list(a.astype(str))
    if arg_plt is None:
        matplotlib.pyplot.title('Epoch: '+str(epoch))
        matplotlib.pyplot.legend(legend_str, loc='upper right')
    else:
        arg_plt.title('Epoch: ' + str(epoch))
        arg_plt.legend(legend_str, loc='upper left', bbox_to_anchor=(1, 1))

    # matplotlib.pyplot.xlim(left=np.min(feat[:, 0]), right=np.max(feat[:, 0]))
    # matplotlib.pyplot.ylim(bottom=np.min(feat[:, 1]), top=np.max(feat[:, 1]))
    # matplotlib.pyplot.text(-7.8,7.3,"epoch=%d" % epoch)

    if not os.path.isdir('images'):
        os.mkdir('images')
    if arg_plt is None:
        matplotlib.pyplot.savefig('./images/epoch=%d.jpg' % epoch)
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()
    else:
        arg_plt.draw()
    # matplotlib.pyplot.pause(0.001)
