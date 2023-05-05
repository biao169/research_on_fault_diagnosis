import numpy as np
import torch


def calculate_Er(ActRUL, RUL):
    er = (ActRUL - RUL) / ActRUL * 100
    er = er.cpu().detach().numpy()
    for i in range(len(er)):
        if er[i] <= 0:
            er[i] = np.exp(-1 * np.log(0.5) + (er[i] / 5))
        else:
            er[i] = np.exp(np.log(0.5) + (er[i] / 20))
    return er


class MultiEi_Loss(torch.nn.Module):
    """ 分类任务损失，目标是使输出神经元尽可能接近1 """

    def __init__(self, num_classes=11):
        super(MultiEi_Loss, self).__init__()
        self.num_classes = num_classes

    def __call__(self, *args, **kwargs):
        if 'num_classes' in kwargs.keys():
            num_classes = kwargs['num_classes']
        else:
            num_classes = self.num_classes
        tmp = None
        for dat in args:
            try:  ## 二维
                batch, num_class = dat.size()
            except:  # 一维
                num_class = 1
            if num_class == 1:
                dat = torch.nn.functional.one_hot(dat, num_classes=num_classes)
            if tmp is None:
                tmp = dat
            else:
                tmp = tmp - dat
        tmp = tmp ** 2
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp
        pass


class Maximum_Mean_Discrepancy(torch.nn.Module):
    """
    MMD Maximum Mean Discrepancy 最大均值差异
    MMD的基本思想就是，如果两个随机变量的任意阶都相同的话，那么两个分布就是一致的。
    而当两个分布不相同的话，那么使得两个分布之间差距最大的那个矩应该被用来作为度量两个分布的标准。
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(Maximum_Mean_Discrepancy, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        pass

    def __call__(self, source, target):
        def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            """计算Gram/核矩阵
            source: sample_size_1 * feature_size 的数据
            target: sample_size_2 * feature_size 的数据
            kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
            kernel_num: 表示的是多核的数量
            fix_sigma: 表示是否使用固定的标准差

                return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                                矩阵，表达形式:
                                [	K_ss K_st
                                    K_ts K_tt ]
            """
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)  # 合并在一起

            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

            # 计算多核中每个核的bandwidth
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

            # 高斯核的公式，exp(-|x-y|/bandwith)
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                          bandwidth_temp in bandwidth_list]

            return sum(kernel_val)  # 将多个核合并在一起

        def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            batch_size = int(source.size()[0])
            kernels = guassian_kernel(source, target,
                                      kernel_mul=self.kernel_mul,
                                      kernel_num=self.kernel_num,
                                      fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]  # Source<->Source
            YY = kernels[batch_size:, batch_size:]  # Target<->Target
            XY = kernels[:batch_size, batch_size:]  # Source<->Target
            YX = kernels[batch_size:, :batch_size]  # Target<->Source
            loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
            # 当不同的时候，就需要乘上上面的M矩阵
            return loss

        return mmd(source, target)




class WGAN_GP(torch.nn.Module):
    """

    """

    def __init__(self, batch_size=128, LAMBDA=0.1, ):
        super(WGAN_GP, self).__init__()
        self.LAMBDA = LAMBDA
        self.alpha = torch.rand([batch_size, 1])
        pass

    def __call__(self, real_data, fake_data, Discriminator):
        alpha = self.alpha.clone().uniform_(0.0, 1.0)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = Discriminator(interpolates)
        # gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradients = torch.autograd.grad(disc_interpolates, [interpolates])[0]
        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
        gradient_penalty = torch.mean((slopes - 1) ** 2)

        return self.LAMBDA * gradient_penalty
