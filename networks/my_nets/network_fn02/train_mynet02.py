import os
import torch
from torch import nn

from train_ways.base import Train_base
from utils_tool.log_utils import Summary_Log, Visual_Model_Predict
from loss_fn.loss_sets import MultiEi_Loss


class Train(Train_base):
    def __init__(self, net:nn.Module, train_dataLoader, test_dataLoader=None, num_classes=10, config:{}=None,  **kwargs):
        super(Train, self).__init__()
        self.train_dataLoader = train_dataLoader
        self.test_dataLoader = test_dataLoader
        self.net = net.to(self.device)
        ''' record the address of train class file'''
        self.opti_log['train way file'] = os.path.abspath(__file__)
        ''' configuration class init. '''
        self.config = config
        self.resultPath = config['resultPath']
        self.epochNum = config['epochNum']
        self.modelPath = os.path.join(self.resultPath, 'model')
        ''' set log mode '''
        log_dict = {'train':['loss', 'acc'], 'test':['loss', 'acc']}
        self.summary_log = Summary_Log(path=self.resultPath, headers=log_dict,
                                       tm_str=self.startTime, **kwargs)
        ''' set loss function '''
        self.loss_fn = MultiEi_Loss(num_classes=num_classes).to(self.device)  #nn.CrossEntropyLoss().to(self.device) ##
        #nn.MSELoss().to(self.device)  ##
        pass

    def training_mode(self, pretrain=False, datasetName='crwu', **kwargs):
        start_epoch = 0
        if pretrain:
            self.net, start_epoch = self.load_model_weight_auto(self.net, **kwargs)
        device = self.device
        net = self.net.to(device)

        optimizer = self.optimizer(net=self.net,
                                   opt_name=str(*self.config['optimizer'].keys()),
                                   kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                   )

        scheduler = self.scheduler(optimizer=optimizer,
                                   sched_name=str(*self.config['scheduler'].keys()),
                                   kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                   )

        """ training system initialization before start training (only one step in advance)"""
        self.training_init_save()
        print('[train]: ================ starting train ========================')
        """ start training """
        # with torch.autograd.set_detect_anomaly(True):
        for idx_epo in range(start_epoch, self.epochNum):
            net.train()
            acc_num = 0
            train_loss = 0
            mini_batch_num = 0
            for i, dat in enumerate(self.train_dataLoader):
                ''' Design program '''
                if datasetName.lower() == 'crwu': de, fe, ba, label = dat[:4]
                elif datasetName.lower() == 'mfpt': label, _, _, de = dat[:4]
                de, label = de.to(device), label.to(device)
                output = net(de)

                loss = self.loss_fn(output, label)  #.squeeze(dim=1)
                pred = output.argmax(dim=1)
                acc_num += torch.eq(pred, label).sum().float().item()
                # print('\t-----', output[:2],  pred[:50], label[:50], acc_num)
                # break

                # if mini_batch_num >= self.config['batchSize']:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                mini_batch_num += len(label)

            torch.cuda.empty_cache()
            train_loss = train_loss / len(self.train_dataLoader)
            train_acc = acc_num / mini_batch_num   ## len(self.train_dataLoader.dataset)
            ''' dynamic adjust lr '''
            try: scheduler.step(train_loss, epoch=None)  ##
            except: scheduler.step(epoch=None)  #  StepLR

            ''' record log ''' # result must the same as the log_dict(above)!
            result = {'loss': train_loss, 'acc': train_acc}
            self.summary_log.add_scalars('train', result, idx_epo, tolerant=True)

            ''' test model online '''
            if (idx_epo+1) % 1 == 0:
                result = self.test_model(net, self.test_dataLoader, pretrain=False, datasetName=datasetName)
                self.summary_log.add_scalars('test', result, idx_epo, tolerant=True)

            ''' 保存模型 '''
            if train_loss < self.summary_log.getMin('train', 'loss', rm_last=True):
                self.saveModel_onlyOne(self.net, idx_epo, 'best')
            if (idx_epo+1)%5==0:
                self.saveModel_onlyOne(self.net, idx_epo, name='')
        pass

    def test_model(self, net=None, dataLoader=None, pretrain=False, datasetName='crwu', **kwargs):
        if net is None: net =self.net
        if pretrain:
            net, epoch = self.load_model_weight_auto(net, **kwargs)
            net = net.to(self.device)
        if dataLoader is None: dataLoader = self.test_dataLoader

        if net is None or dataLoader is None:
            res = {}
            for k in self.summary_log.headers['test']:
                res[k] = 0
            return res
        device = self.device
        torch.cuda.empty_cache()
        net.eval()
        acc_num = 0
        train_loss = 0
        mini_batch_num = 0
        for i, dat in enumerate(dataLoader):
            ''' Design program '''
            if datasetName.lower() == 'crwu': de, fe, ba, label = dat[:4]
            elif datasetName.lower() == 'mfpt': label, _, _, de = dat[:4]
            de, label = de.to(device), label.to(device)
            output = net(de)

            loss = self.loss_fn(output, label)  # .squeeze(dim=1)
            pred = output.argmax(dim=1)

            acc_num += torch.eq(pred, label).sum().float().item()
            train_loss += float(loss.item())
            mini_batch_num += len(label)

        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)
        return {'loss': train_loss, 'acc': train_acc}

    def test_model_visualization(self, net=None, dataLoader=None, pretrain=False, datasetName='crwu', **kwargs):
        if net is None: net = self.net
        if pretrain:
            net, epoch = self.load_model_weight_auto(net, **kwargs)
            net = net.to(self.device)
        if dataLoader is None: dataLoader = self.test_dataLoader

        if net is None or dataLoader is None:
            res = {}
            for k in self.summary_log.headers['test']:
                res[k] = 0
            return res
        """ Visual """
        vs = Visual_Model_Predict()

        device = self.device
        torch.cuda.empty_cache()
        net.eval()
        acc_num = 0
        train_loss = 0
        mini_batch_num = 0
        for i, dat in enumerate(dataLoader):
            ''' Design program '''
            if datasetName.lower() == 'crwu': de, fe, ba, label = dat[:4]
            elif datasetName.lower() == 'mfpt': label, _, _, de = dat[:4]
            de, label = de.to(device), label.to(device)
            output = net(de)
            # print('--------', de.shape)
            # exit()
            # print('==',output, label, output.argmax(dim=1) )
            loss = self.loss_fn(output, label)
            pred = output.argmax(dim=1)
            acc_num += torch.eq(pred, label).sum().float().item()
            mini_batch_num += len(label)

            train_loss += float(loss.item())

            vs.add_data_series(data={'label': label.detach().cpu().numpy(), 'predict': pred.detach().cpu().numpy()})
            # print({'label': rate.detach().cpu().numpy(), 'predict': conf.detach().cpu().numpy()})
            # break
        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)

        ''' record log '''  # result must the same as the log_dict(above)!
        result = {'loss': train_loss, 'acc': train_acc}
        print('test model：', result)
        vs.draw_figure_matrix(keys=['label', 'predict'])
        return result




