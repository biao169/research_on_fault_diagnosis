import numpy as np
import torch
def calculate_Er(ActRUL, RUL):
    er = (ActRUL-RUL)/ActRUL*100
    er = er.cpu().detach().numpy()
    for i in range(len(er)):
        if er[i]<=0:
            er[i] = np.exp(-1*np.log(0.5)+(er[i]/5))
        else:
            er[i] = np.exp(np.log(0.5)+(er[i]/20))
    return er



class MultiEi_Loss(torch.nn.Module):
    """ 分类任务损失，目标是使输出神经元尽可能接近1 """
    def __init__(self, num_classes=11):
        super(MultiEi_Loss, self).__init__()
        self.num_classes = num_classes
    def __call__(self, *args, **kwargs):
        if 'num_classes' in kwargs.keys():
            num_classes = kwargs['num_classes']
        else :num_classes = self.num_classes
        tmp = None
        for dat in args:
            try:  ## 二维
                batch, num_class = dat.size()
            except: #  一维
                num_class =1
            if num_class ==1:
                dat = torch.nn.functional.one_hot(dat, num_classes=num_classes)
            if tmp is None: tmp=dat
            else: tmp = tmp-dat
        tmp = tmp**2
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp
        pass

