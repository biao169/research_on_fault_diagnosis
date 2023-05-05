import random
import torch
import torch.utils
from torch.utils import data
import numpy as np
import os

from dataset.dataset_CRWU import DataSet_CRWU_MultiFile as DataSet_MFPT_MultiFile
# from dataset.dataset_KAT import DataSet_KAT_MultiFile as DataSet_MFPT_MultiFile
# from dataset.dataset_DNU import DataSet_DNU_MultiFile as DataSet_MFPT_MultiFile
# from dataset.dataset_MFPT import DataSet_MFPT_MultiFile
from networks.my_nets.network_fn03 import mynet03 as Net
import networks.my_nets.network_fn03.train_mynet03 as Train
from utils_tool.log_utils import YamlHandler, delete_all_log_record_file

from report_make.excel_report import Make_Report


config = YamlHandler('config.yaml').read_yaml()


# print(config)
# exit()
def seed_torch(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
seed_torch(config['seed'])

datasetName = 'CRWU'  #'MFPT' #
train_set = 'crwu_set1'  #'mfpt_set1'  #
num_class = len(config['data_resource'][datasetName][train_set])  #

def dcnn_config(config=config, set='mfpt_set1'):
    """ 选优样本 """
    root = config['data_resource'][datasetName]['path']
    config['train_loader']['dataset']['path'] = root
    config['test_loader']['dataset']['path'] = root
    load_files = config['data_resource'][datasetName][set]  ## 'mfpt_set1'

    config['batchSize'] = 32
    config['train_loader']['dataset']['miniBatch'] = 256
    config['test_loader']['dataset']['miniBatch'] = 256

    config['optimizer'].pop('SGD')
    # config['optimizer']['Adam'] = dict(lr=0.001, betas=[0.9, 0.999], weight_decay=0.0001)
    config['optimizer']['SGD'] = dict(lr=0.01, momentum=0.9, weight_decay=0.001)
    config['scheduler']['ReduceLROnPlateau'] = dict(mode='min', factor=0.1, patience=10, eps=0.00000000001, cooldown=5)
    # config['scheduler']['StepLR'] = dict(step_size=40, gamma=0.1, last_epoch=-1)
    return config.copy(), load_files


def dcnn_config2(config=config, set='mfpt_set1'):
    """ Only for evaluation in another condition or fault """
    load_files = config['data_resource'][datasetName][set]
    config, _ = dcnn_config()
    return config, load_files


def dcnn_config_kat(config=config):
    """ Only for evaluation in another condition or fault """
    load_files = [
        [r'KA01\N09_M07_F10_KA01_1.mat', r'KA01\N09_M07_F10_KA01_2.mat', r'KA01\N09_M07_F10_KA01_3.mat', ],
        [r'KA03\N09_M07_F10_KA03_1.mat', r'KA03\N09_M07_F10_KA03_2.mat', r'KA03\N09_M07_F10_KA03_3.mat', ],
        [r'KA05\N09_M07_F10_KA05_1.mat', r'KA05\N09_M07_F10_KA05_2.mat', r'KA05\N09_M07_F10_KA05_3.mat', ],
        [r'KA06\N09_M07_F10_KA06_1.mat', r'KA06\N09_M07_F10_KA06_2.mat', r'KA06\N09_M07_F10_KA06_3.mat', ],
        [r'KA07\N09_M07_F10_KA07_1.mat', r'KA07\N09_M07_F10_KA07_2.mat', r'KA07\N09_M07_F10_KA07_3.mat', ],
        [r'KA08\N09_M07_F10_KA08_1.mat', r'KA08\N09_M07_F10_KA08_2.mat', r'KA08\N09_M07_F10_KA08_3.mat', ],
        [r'KA09\N09_M07_F10_KA09_1.mat', r'KA09\N09_M07_F10_KA09_2.mat', r'KA09\N09_M07_F10_KA09_3.mat', ],
        [r'KB23\N09_M07_F10_KB23_1.mat', r'KB23\N09_M07_F10_KB23_2.mat', r'KB23\N09_M07_F10_KB23_3.mat', ],
        [r'KI01\N09_M07_F10_KI01_1.mat', r'KI01\N09_M07_F10_KI01_2.mat', r'KI01\N09_M07_F10_KI01_3.mat', ],
        [r'KI03\N09_M07_F10_KI03_1.mat', r'KI03\N09_M07_F10_KI03_2.mat', r'KI03\N09_M07_F10_KI03_3.mat', ],
    ]
    config, _ = dcnn_config()
    config['train_loader']['dataset']['path'] = r'I:\python_datasets\PU dataset'
    config['test_loader']['dataset']['path'] = r'I:\python_datasets\PU dataset'
    return config.copy(), load_files


def dcnn_config_DNU(config=config):
    """ Only for evaluation in another condition or fault """
    load_files = [
        r'ball_20_0.csv',
        r'ball_30_2.csv',
        r'comb_20_0.csv',
        r'comb_30_2.csv',
        r'inner_20_0.csv',
        r'inner_30_2.csv',
        r'outer_20_0.csv',
        r'outer_30_2.csv',
        r'health_20_0.csv',
        r'health_30_2.csv',
    ]
    config, _ = dcnn_config()
    config['train_loader']['dataset']['path'] = r'I:\python_datasets\Mechanical-datasets-master\gearbox\bearingset'
    config['test_loader']['dataset']['path'] = r'I:\python_datasets\Mechanical-datasets-master\gearbox\bearingset'
    return config.copy(), load_files


def designed_train():
    # Fixme
    net = Net.DRFNs_Predict(in_channel=1, out_channel=num_class, net='mf')
    config, load_files = dcnn_config()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_designed'.format(net.work_name))

    # train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered(
    #                     config['train_loader']['dataset']['path'], load_files,
    #                     train_test_rate=0.7, repeat_win=0.85, window=2048)
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.7, repeat_win=0.85, window=2048,
        mask_win=int(2048 * 0),  # int(2048*0.4)
        mask_start_idx=None,
        snr=None)

    print(f'train_dataset len = {len(train_dataset)},\t test_dataset len = {len(test_dataset)}')

    trainloader = data.DataLoader(dataset=train_dataset, batch_size=config['train_loader']['dataset']['miniBatch'],
                                  shuffle=config['train_loader']['shuffle'], drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=config['test_loader']['dataset']['miniBatch'],
                                 shuffle=config['test_loader']['shuffle'], drop_last=False, )
    
    """ 删除日志文件 """
    delete_all_log_record_file(root=config['resultPath'], idx=[2], tm=None, rm_config=True, rm_csv=True,
                               rm_model=True, rm_tensor=True)
    config['epochNum'] = 1000
    train = Train.Train(net=net, train_dataLoader=trainloader, test_dataLoader=testloader, num_classes=num_class,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',  ## 'loss-acc'  'train-test'
                        new_thread=True)
    train.training_mode(pretrain=False, datasetName=datasetName)
    exit()


def model_evaluation(mask_win=0,
                     mask_start_idx=0,
                     snr=10, model=None, set='crwu_set1_2', is_auto=False):
    # Fixme
    net = Net.DRFNs_Predict(in_channel=1, out_channel=num_class, net='mf')
    config, load_files = dcnn_config(set=set)
    # config, load_files = dcnn_config_kat()
    # config, load_files = dcnn_config_DNU()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_designed'.format(net.work_name))

    """ 测试网络 """
    # result = train.test_model(dataLoader=testloader, pretrain=True,
    #                           file='checkpoint_2022-10-10 17.56.05_epo[1724].model'
    #                           )
    # print('test model:', result)

    print(""" --------------------------------------------- """)
    if not is_auto:
        mask_win = 0.2
        mask_start_idx = 0.4
        snr = 4
        if model==None: model = 'checkpoint_2022-10-28 13.34.13_epo[553]_best.model'
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset = DataSet_KAT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset=DataSet_DNU_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.7, repeat_win=0.85, window=2048,
        mask_win=int(2048 * mask_win),
        mask_start_idx=int(2048 * mask_start_idx),
        snr=snr)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    print('orignal size={}  new test_dataset size={}'.format(len(train_dataset), len(test_dataset)))

    testloader = data.DataLoader(dataset=test_dataset, batch_size=512,
                                 shuffle=False, drop_last=False, )
    print('new test_dataloader size={}'.format(len(testloader)))

    train = Train.Train(net=net, train_dataLoader=None, test_dataLoader=testloader, num_classes=num_class,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                        new_thread=True)

    result = train.test_model_visualization(dataLoader=testloader, pretrain=True, datasetName=datasetName,
                                            file=model
                                            )
    print('test model:', result)

    from matplotlib import pyplot as plt
    plt.title(f'mask size={mask_win}  mask idx={mask_start_idx}  SNR={snr}')
    plt.draw()
    plt.pause(0.1)
    if not is_auto:
        plt.show()
    else:
        path = config['resultPath'] + '\\' + r'matrix img'
        file = 'matrix mask{} maskidx{} snr{}.png'.format(mask_win, mask_start_idx, snr)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, file)
        print('save fig at:', path)
        plt.savefig(path)
    return result['acc']
    # plt.show()


def t_sne_vishow(mask_win=0, mask_start_idx=0, snr=0, model='', set='', is_auto=False, isave=False):
    from utils_tool.t_SNE import t_SNE_Visualization as T_SNE
    # Fixme
    net = Net.DRFNs_Predict(in_channel=1, out_channel=num_class, net='gf')
    config, load_files = dcnn_config()
    if not (set is None or set == ''): load_files = config[set]
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_designed'.format(net.work_name))
    # print(net)
    """ 测试网络 """
    tsne = T_SNE()
    tsne.set_hook(net.fc, 3)

    print(""" ============================================================ """)
    if not is_auto:
        mask_win = 0
        mask_start_idx = 0
        snr = 0
        model = 'checkpoint_2022-11-06 13.00.59_epo[653]_best.model'
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset = DataSet_KAT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset=DataSet_DNU_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.5, repeat_win=0.85, window=2048,
        mask_win=int(2048 * mask_win),
        mask_start_idx=int(2048 * mask_start_idx),
        snr=snr)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    print('orignal size={}  new test_dataset size={}'.format(len(train_dataset), len(test_dataset)))

    testloader = data.DataLoader(dataset=test_dataset, batch_size=512,
                                 shuffle=False, drop_last=False, )
    print('new test_dataloader size={}'.format(len(testloader)))

    train = Train.Train(net=net, train_dataLoader=None, test_dataLoader=testloader,num_classes=num_class,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                        new_thread=True)
    net, epoch = train.load_model_weight_auto(net, file=model)
    for i, dat in enumerate(testloader):
        ''' Design program '''
        if datasetName.lower() == 'crwu':
            de, fe, ba, label = dat[:4]
        elif datasetName.lower() == 'mfpt':
            label, _, _, de = dat[:4]
        de, label = de.to(train.device), label.to(train.device)
        output = net(de)
        tsne.add_label(label)

    tsne.t_sne_plot()
    from matplotlib import pyplot as plt
    plt.title(f'net={net.log}  mask size={mask_win}  mask idx={mask_start_idx}  SNR={snr}')
    plt.draw()
    plt.pause(0.1)

    if isave:
        path = config['resultPath'] + '\\' + r'matrix img'
        file = 't-sne {} matrix mask{} maskidx{} snr{}.png'.format(net.log, mask_win, mask_start_idx, snr)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, file)
        print('save fig at:', path)
        plt.savefig(path)

    plt.show()
    pass


def run():
    """ 数据处理，加噪声 """
    # designed_train()

    """ 模型性能测试 """
    # model_evaluation(set='mfpt_set1', model = 'checkpoint_2023-02-21 09.37.35_epo[861]_best.model')
    # t_sne_vishow(isave=True)
    # exit()

    mask_size = [0, 0.2, 0.4, 0.6]  # , 0.8
    mask_idx = [0, 0.2, 0.4, 0.6, 0.8]
    snr = [-4, -2, 0, 2, 4, 6, 8, 10]

    model =  'checkpoint_2023-04-02 18.00.49_epo[266]_best.model'
    eval_set = train_set  #  'mfpt_set1' #'crwu_set1'  #
    train_way = dict(
        a=dict(network='drfn-auto', dataset='mfpt_set1', train_size=2048, evalute_szie=2048),
        b=dict(mask='0', mask_idx='0', SNR='random', eval_set=eval_set, kernel=7),  # random  #  0
    )
    dd = dict(save_file=os.path.join(config['resultPath'], 'drfn_record.xlsx'),
              model_path=os.path.join(config['resultPath'], 'DRFNs_designed', 'model', model),  # Net2_Predict_designed
              train_way=train_way,
              start_row=1 + 23 * 3, sheet=0,
              )
    report = Make_Report(**dd)
    for ms in mask_size:
        for mi in mask_idx:
            for sn in snr:
                print('\n', '='*20, ' '*3, 'running:', ms, '---mask idx:', mi, '---SNR:', sn, ' '*3, '='*20, )
                acc = model_evaluation(mask_win=ms, mask_start_idx=mi, snr=sn, model=model, set=eval_set, is_auto=True)
                report.add_data(mask_size=ms, mask_idx=mi, snr=sn, acc=acc)  #

            if ms + mi >= 1 or ms == 0: break
    try:
        report.record_data_in_excel()
    except:
        print('\033[41m 快关闭数据文件，要保存啦!!')
        x = input('是否保存【y/n or 1/0】：')
        if not (str(x).lower()=='n' or str(x).lower()=='0'):
            report.record_data_in_excel()
            print('数据文件保存完成！ \033[0m ')
        else: print('不保存  \033[0m ')


if __name__ == '__main__':
    run()
