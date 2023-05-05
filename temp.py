import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# import datetime
# import threading

# from dataset import dataset_CRWU, dataset_MFPT
#
# from torch.utils.tensorboard import SummaryWriter
# from utils_tool.log_utils import delete_path, Visual_Model_Predict, YamlHandler
# from report_make import excel_report as report
#

from dataset.dataset_MFPT import DataSet_MFPT_MultiFile

train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
    root=config['train_loader']['dataset']['path'],
    load_files=load_files,
    train_test_rate=0.7, repeat_win=0.85, window=2048,
    mask_win=int(2048 * 0),  # int(2048*0.4)
    mask_start_idx=None,
    snr=None)





exit()


def calculator_outSize(insize, kernel, padding, stride):
    """ 计算卷积层输出尺寸 """
    return (insize - kernel + 2 * padding) // stride + 1


def calculator_padding(insize, outsize, kernel, stride):
    """ 计算需要的padding 尺寸  """
    return ((outsize - 1) * stride - insize + kernel) / 2


d = calculator_padding(32, 64, 64, 16)
print(d)
d = calculator_outSize(32, 5, 2, 1)
print(d)

exit()

dd = dict(save_file=r'I:\python\01-work\result\aa.xlsx',
          model_path=r'I:\python\01-work\result\aaa.model',
          network='drsn', dataset='crwu', train_size=2048, evalute_szie=2048,
          SNR='random', mask='random', mask_idx='random', snr='random',
          start_row=1 + 23 * 0, new_sheet=False,
          )
re = report.Make_Report(**dd)
re.add_data(0.2, 0.4, -4, 0.35)
re.add_data(0.2, 0.4, 0, 0.9785)
re.add_data(0.6, 0.2, 0, 0.46)
re.record_data_in_excel()
exit()

vs = Visual_Model_Predict()
vs.draw_figure_matrix()

exit()

# dataset_CRWU.run()


exit()


def calculate_days(tm_str, tm_str2):
    tm = tm_str + ' 12:00:00'
    start = datetime.datetime.strptime(tm, '%Y-%m-%d %H:%M:%S')
    tm = tm_str2 + ' 12:00:00'
    end = datetime.datetime.strptime(tm, '%Y-%m-%d %H:%M:%S')
    days = (end - start).days
    # print(days)
    return days


lpr = {'2021-11-22': 3.85, '2021-12-20': 3.80, '2022-01-20': 3.70, '2022-02-21': 3.70, '2022-03-21': 3.70,
       '2022-04-20': 3.70, '2022-05-20': 3.70, '2022-06-20': 3.70, '2022-07-20': 3.70, '2022-08-22': 3.65,
       '2022-09-20': 3.65, '2022-10-20': 3.65, }

money = 35000
m = 2
year_days = 365
interest_s = []
start = '2021-12-01'
interest = calculate_days('2021-12-01', '2021-12-20') * lpr['2021-11-22'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2021-12-20', '2022-01-20') * lpr['2021-12-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-01-20', '2022-02-21') * lpr['2022-01-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-02-21', '2022-03-21') * lpr['2022-02-21'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-03-21', '2022-04-20') * lpr['2022-03-21'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-04-20', '2022-05-20') * lpr['2022-04-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-05-20', '2022-06-20') * lpr['2022-05-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-06-20', '2022-07-20') * lpr['2022-06-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-07-20', '2022-08-22') * lpr['2022-07-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-08-22', '2022-09-20') * lpr['2022-08-22'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-09-20', '2022-10-20') * lpr['2022-09-20'] / year_days * m * money
interest_s.append(interest)
interest = calculate_days('2022-10-20', '2022-10-28') * lpr['2022-10-20'] / year_days * m * money
interest_s.append(interest)
interest_s = np.array(interest_s) / 100
print('每个月的利息为：', interest_s)
print('总利息为：', interest_s.sum())

# calculate_days('2021-11-22', '2021-12-20')


exit()

# file = r'E:\桌面文件\08-科研辅助-事务\03-万老师课题组资料-事务\万-个人资料\万老师的工作成绩附件\万加富2018以后的SCI论文情况.xlsx'
# excel = pd.read_excel(file, header=None)
# num = len(excel)
# print(num)
# first_num = 0
# co_author_num = 0
# author_num = 0
# for k in range(int(num/3)):
#     i = k*3
#     paper = str(excel[0][i]).lower()
#     author = str(excel[0][i+1]).lower()
#     journal = str(excel[0][i+2]).lower()
#     # print(paper, author, journal, '\n')
#     # print(k+1)
#     if '2018.' in journal or '2019.' in journal or \
#         '2020.' in journal or '2021.' in journal or \
#         '2022.' in journal or '2023.' in journal:
#         # print('==', i+1)
#         if author.startswith('jiafu wan'): first_num+=1
#         if 'corresponding' in author: co_author_num+=1
#         if 'jiafu wan' in author: author_num+=1
#
#     pass
# print('first_num=', first_num )
# print('co_author_num=', co_author_num)
# print('author_num=', author_num)
