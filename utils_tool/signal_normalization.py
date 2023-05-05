import os
import numpy as np
import pywt
# from scipy.fftpack import fft, ifft  ## 傅里叶变换


def fft_transform(x):
    """ 快速傅里叶变换：时频转换 """
    n = len(x)
    y = np.fft.fft(x)/n
    return abs(y)  #np.sqrt(np.real(y)**2 +np.imag(y)**2)   # ## 取实数部分


# totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
def TimeFrequencyCWT(data, fs, totalscal, wavelet='cgau8'):
    """ 连续小波傅里叶变换 """
    # 采样数据的时间维度
    # t = np.arange(data.shape[0]) / fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0 / fs)
    return [cwtmatr, frequencies]


from io import BytesIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv
def Wavelet_Fourier_transform_numpy(data):
    """ 连续小波傅里叶变换 --在线转 numpy """
    [cwtmatr, frequencies] = TimeFrequencyCWT(data, len(data), totalscal=256, wavelet='cgau8')

    t = np.arange(data.shape[0]) / len(data)
    plt.figure(figsize=(2, 2))
    plt.clf()
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.subplots_adjust(top=1, left=0, right=1, bottom=0)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    new_img = np.asarray(Image.open(buffer), dtype=np.uint8)  # new_img就是figure的数组
    plt.close()
    buffer.close()
    # print('Wavelet_Fourier_transform_numpy 1:', new_img.shape)
    new_img = cv.cvtColor(new_img, cv.COLOR_BGRA2GRAY)
    # print('Wavelet_Fourier_transform_numpy 3:', new_img.shape)
    return new_img
    pass








def add_gauss_noise(x, snr):
    """ 加入高斯噪声 """
    # print('add_gauss_noise:', x.shape, len(x))
    if snr==0: return x
    Ps = np.sum(abs(x)**2)/len(x)
    # Pn = Ps/(10**((snr/10)))
    Pn = Ps/np.power(10, (snr/10))  #计算噪声设定的方差值
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

def signal_mask_random(x, start_idx:int=None, mask_win:int=None):
    """ 添加（可随机）掩模，使信号部分为0，模拟信号突发性丢失场景
        :return x, rate
    """
    length = len(x)
    # if mask_win is None:
    #     mask_win = np.random.randint(0, length)
    # if start_idx is None:
    #     start_idx = np.random.randint(0, length-mask_win)
    if mask_win == 0: return x, 1.0, 1.0
    rate = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if mask_win is None:
        if start_idx is not None:
            m = round(1 - start_idx / length, 2)
            rate = rate[rate<=m]
        idx = np.random.randint(0, len(rate))
        mask_win = int( rate[idx ]*length )
        # print( 'signal_mask_random:', idx, mask_win, start_idx)

    if start_idx is None:
        max_start_rate = float(str(1-mask_win/length)[:3])  ## 保留1位小数，避免值大于实际，不用四舍五入
        if max_start_rate >0:
            start_idx_rate = np.arange(0, max_start_rate, 0.1)
            start_idx = int(start_idx_rate[np.random.randint(0, len(start_idx_rate))] * length)
        else: start_idx = 0

    if start_idx+mask_win > length:
        raise ValueError(f'[start_idx+mask_win] can not > {length}! your start_idx={start_idx}, mask_win={mask_win}')
    if mask_win >0: x[start_idx: start_idx+mask_win] = 0
    rate = round(1-mask_win/length, 2)
    # print('signal_mask_random:', rate, mask_win, start_idx)
    return x, rate, round(start_idx/length, 2)



