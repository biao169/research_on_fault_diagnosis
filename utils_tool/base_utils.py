
class PrintShow:
    """ 配置打印输出屏幕的颜色 """
    # color = dict(
    defualt='\033[0m',
    error='\033[31m',
    warning='\033[33m',
    tip_green='\033[32m',
    tip_blue='\033[34m',
    tip_cyan='\033[36m',
    tip_purple='\033[35m',
    tip_white='\033[30m',
    tip_b_white='\033[07m',
    tip_bm_white='\033[7;30m',
    tip_b_red='\033[7;31;40mm',
    tip_b_green='\033[7;32;40m',
    tip_b_blue='\033[7;34;40m',
    tip_b_cyan='\033[7;36;40m',
    tip_b_purple='\033[7;35;40m',
    tip_label_white = '\033[51m',
    # )
    def __init__(self):

        # color2 = dict(
        #     defualt='\033[0m',
        #     error='\033[0;31;40m',
        #     warning='\033[0;33;40m',
        #     tip_green='\033[0;32;40m',
        #     tip_blue='\033[0;34;40m',
        #     tip_cyan='\033[0;36;40m',
        #     # ,
        # )
        pass



def calculator_outSize(insize, kernel, padding, stride):
    """ 计算卷积层输出尺寸 """
    return (insize - kernel + 2 * padding) // stride + 1


def calculator_padding(insize, outsize, kernel, stride):
    """ 计算需要的padding 尺寸  """
    return ((outsize - 1) * stride - insize + kernel) / 2



