import os
import torch
from torch import nn

from train_ways.base import Train_base
from utils_tool.log_utils import Summary_Log, Visual_Model_Predict
from loss_fn.loss_sets import Maximum_Mean_Discrepancy
from networks.sota_nets.n02_GNN_ODA.gnn_oda import GNN_ODA


class Train(Train_base):
    def __init__(self, net: nn.Module, train_dataLoader, test_dataLoader=None, config: {} = None, **kwargs):
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
        result = ['loss', 'acc', 'mmd-loss', 'real-loss', 'fake-loss']
        log_dict = {'train': result, 'test': result}
        self.summary_log = Summary_Log(path=self.resultPath, headers=log_dict,
                                       tm_str=self.startTime, **kwargs)
        ''' set loss function '''
        self.loss_fn = Maximum_Mean_Discrepancy(kernel_mul=2.0, kernel_num=5, fix_sigma=None).to(
            self.device)  ## nn.MSELoss().to(self.device)
        self.loss_fn_classify = nn.CrossEntropyLoss()
        pass

    def format_train(self, dat, net: GNN_ODA, datasetName, device):
        if datasetName.lower() == 'crwu':
            label, _, _, de, fe, ba = dat[:6]
        elif datasetName.lower() == 'mfpt':
            label, _, fe, de = dat[:4]
        de, fe, label = de.to(device), fe.to(device), label.to(device)
        extractor, generator, real_out, fake_out = net(de)
        output = net(de)
        mmdloss = self.loss_fn(extractor, generator)
        real_croloss = self.loss_fn_classify(real_out, label)
        fake_croloss = self.loss_fn_classify(fake_out, label)

        loss = mmdloss + real_croloss + fake_croloss
        pred = output.argmax(dim=1)
        eq_num = torch.eq(pred, label).sum().float().item()
        return loss, eq_num, label, output, mmdloss, real_croloss, fake_croloss

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
        for idx_epo in range(start_epoch, self.epochNum):
            net.train()
            acc_num = 0
            train_loss = 0
            mini_batch_num = 0
            epoch_mmdloss, epoch_real_croloss, epoch_fake_croloss = 0, 0, 0

            for i, dat in enumerate(self.train_dataLoader):
                ''' Design program '''
                # if datasetName.lower() == 'crwu': de, fe, ba, label = dat[:4]
                # elif datasetName.lower() == 'mfpt': label, _, fe, de = dat[:4]
                # de, fe, label = de.to(device), fe.to(device), label.to(device)
                # x, x2, out1, out2 = net(de)
                # output = net(de)
                # loss = self.loss_fn(output, label)  #.squeeze(dim=1)
                # pred = output.argmax(dim=1)
                loss, eq_num, label, output, mmdloss, real_croloss, fake_croloss = \
                    self.format_train(dat, net, datasetName, device)

                acc_num += eq_num
                epoch_mmdloss += float(mmdloss.item())
                epoch_real_croloss += float(real_croloss.item())
                epoch_fake_croloss += float(fake_croloss.item())

                # if mini_batch_num >= self.config['batchSize']:
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                optimizer.zero_grad()
                train_loss += float(loss.item())
                mini_batch_num += len(label)

            torch.cuda.empty_cache()
            train_loss = train_loss / len(self.train_dataLoader)
            train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)
            ''' dynamic adjust lr '''
            scheduler.step(train_loss)

            ''' record log '''  # result must the same as the log_dict(above)!
            result = {'loss': train_loss, 'acc': train_acc, 'mmd-loss': epoch_mmdloss,
                      'real-loss': epoch_real_croloss, 'fake-loss': epoch_fake_croloss}
            self.summary_log.add_scalars('train', result, idx_epo, tolerant=True)

            ''' test model online '''
            if (idx_epo + 1) % 1 == 0:
                result = self.test_model(net, self.test_dataLoader, pretrain=False, datasetName=datasetName)
                self.summary_log.add_scalars('test', result, idx_epo, tolerant=True)

            ''' 保存模型 '''
            if train_loss < self.summary_log.getMin('train', 'loss', rm_last=True):
                self.saveModel_onlyOne(self.net, idx_epo, 'best')
            if (idx_epo + 1) % 5 == 0:
                self.saveModel_onlyOne(self.net, idx_epo, name='')
        pass

    def test_model(self, net=None, dataLoader=None, pretrain=False, datasetName='crwu', **kwargs):
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
        device = self.device
        torch.cuda.empty_cache()
        net.eval()
        acc_num = 0
        train_loss = 0
        mini_batch_num = 0
        epoch_mmdloss, epoch_real_croloss, epoch_fake_croloss = 0, 0, 0
        for i, dat in enumerate(dataLoader):
            ''' Design program '''
            loss, eq_num, label, output, mmdloss, real_croloss, fake_croloss = \
                self.format_train(dat, net, datasetName, device)

            acc_num += eq_num
            train_loss += float(loss.item())
            mini_batch_num += len(label)

            epoch_mmdloss += float(mmdloss.item())
            epoch_real_croloss += float(real_croloss.item())
            epoch_fake_croloss += float(fake_croloss.item())

        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)
        result = {'loss': train_loss, 'acc': train_acc, 'mmd-loss': epoch_mmdloss,
                  'real-loss': epoch_real_croloss, 'fake-loss': epoch_fake_croloss}
        return result

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

        epoch_mmdloss, epoch_real_croloss, epoch_fake_croloss = 0, 0, 0
        for i, dat in enumerate(dataLoader):
            ''' Design program '''
            loss, eq_num, label, output, mmdloss, real_croloss, fake_croloss = \
                self.format_train(dat, net, datasetName, device)

            acc_num += eq_num
            mini_batch_num += len(label)
            train_loss += float(loss.item())

            epoch_mmdloss += float(mmdloss.item())
            epoch_real_croloss += float(real_croloss.item())
            epoch_fake_croloss += float(fake_croloss.item())

            vs.add_data_series(
                data={'label': label.detach().cpu().numpy(), 'predict': output.argmax(dim=1).detach().cpu().numpy()})
            # print({'label': rate.detach().cpu().numpy(), 'predict': conf.detach().cpu().numpy()})
            # break
        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)

        ''' record log '''  # result must the same as the log_dict(above)!
        result = {'loss': train_loss, 'acc': train_acc, 'mmd-loss': epoch_mmdloss,
                  'real-loss': epoch_real_croloss, 'fake-loss': epoch_fake_croloss}
        print('test model：', result)
        vs.draw_figure_matrix(keys=['label', 'predict'])
        return result
