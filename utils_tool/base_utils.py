

def calculator_outSize(insize, kernel, padding, stride):
    """ 计算卷积层输出尺寸 """
    return (insize - kernel + 2 * padding) // stride + 1


def calculator_padding(insize, outsize, kernel, stride):
    """ 计算需要的padding 尺寸  """
    return ((outsize - 1) * stride - insize + kernel) / 2



