import os
import torch
from torch import nn

from train_ways.base import Train_base
from utils_tool.log_utils import Summary_Log, Visual_Model_Predict
from loss_fn.loss_sets import Maximum_Mean_Discrepancy
from networks.sota_nets.n04_DGGP_ISVDD.dggp_isvdd import Generate_E, Generate_S, Discriminator_E, Discriminator_S


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
        result = ['loss', 'disc-loss-E', 'disc-loss-S', 'gene-loss-ES']
        log_dict = {'train': result, 'test': result}
        self.summary_log = Summary_Log(path=self.resultPath, headers=log_dict,
                                       tm_str=self.startTime, **kwargs)
        ''' set loss function '''
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)  ## nn.MSELoss().to(self.device)

        pass

    def deploy_nets(self, generate_E, generate_S, discriminator_E, discriminator_S, batchSize=128):
        self.generate_E = generate_E.to(self.device)
        self.generate_S = generate_S.to(self.device)
        self.discriminator_E = discriminator_E.to(self.device)
        self.discriminator_S = discriminator_S.to(self.device)
        # self.batchSize = batchSize
        assert batchSize%2==0, 'batch size should be a multiple of 2!'
        self.optimizer_GE = self.optimizer(net=generate_E,
                                           opt_name=str(*self.config['optimizer'].keys()),
                                           kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                           )
        self.scheduler_GE = self.scheduler(optimizer=self.optimizer_GE,
                                           sched_name=str(*self.config['scheduler'].keys()),
                                           kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                           )
        self.optimizer_GS = self.optimizer(net=generate_S,
                                           opt_name=str(*self.config['optimizer'].keys()),
                                           kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                           )
        self.scheduler_GS = self.scheduler(optimizer=self.optimizer_GS,
                                           sched_name=str(*self.config['scheduler'].keys()),
                                           kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                           )
        self.optimizer_DE = self.optimizer(net=discriminator_E,
                                           opt_name=str(*self.config['optimizer'].keys()),
                                           kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                           )
        self.scheduler_DS = self.scheduler(optimizer=self.optimizer_DE,
                                           sched_name=str(*self.config['scheduler'].keys()),
                                           kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                           )
        self.optimizer_DS = self.optimizer(net=discriminator_S,
                                           opt_name=str(*self.config['optimizer'].keys()),
                                           kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                           )
        self.scheduler_DS = self.scheduler(optimizer=self.optimizer_DS,
                                           sched_name=str(*self.config['scheduler'].keys()),
                                           kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                           )

        pass

    def format_train(self, dat, net: torch.nn.Module, datasetName, device):
        if datasetName.lower() == 'crwu':
            label, _, _, de, fe, ba = dat[:6]
        elif datasetName.lower() == 'mfpt':
            label, _, fe, de = dat[:4]
        de, fe, label = de.to(device), fe.to(device), label.to(device)
        batch_size = de.shape[0] // 2
        real_fore_dat, real_back_dat = de[:batch_size], de[batch_size:]
        # 第一重1
        fake_back_data = self.generate_E(real_fore_dat)
        # print('fake_back_data:', fake_back_data.shape, real_back_dat.shape)
        out_discriminator_E_fake = self.discriminator_E(fake_back_data)
        loss_discriminator_e = out_discriminator_E_fake - self.discriminator_E(real_back_dat)  # 这个越小， 生成器越好
        # 第一重2
        fake_fore_data = self.generate_S(real_back_dat)
        out_discriminator_S_fake = self.discriminator_S(fake_fore_data)
        loss_discriminator_s = out_discriminator_S_fake - self.discriminator_S(real_fore_dat)  # 这个越小， 生成器越好

        # 循环对抗: 第二重
        lambda_E, lambda_S = 0.5, 0.5
        loss_generate_e = lambda_E * torch.sqrt((real_fore_dat - self.generate_S(fake_back_data)) ** 2)
        loss_generate_s = lambda_S * torch.sqrt((real_back_dat - self.generate_E(fake_fore_data)) ** 2)
        loss_Ge_Gs = loss_generate_e + loss_generate_s - out_discriminator_E_fake - out_discriminator_S_fake

        ''' 在GAN中，使用梯度惩罚来收敛  WGAN的内容'''
        def loss_GP(real_data, fake_data, Discriminator):
            alpha = torch.rand([batch_size, 1, 1]).uniform_(0.0, 1.0).cuda()
            interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            # print('loss_GP:', alpha.shape, interpolates.shape )
            disc_interpolates = Discriminator(interpolates)
            # gradients = tf.gradients(disc_interpolates, [interpolates])[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

            gradients = torch.autograd.grad(disc_interpolates, [interpolates],
                                            grad_outputs=torch.ones_like(disc_interpolates).cuda(),
                                            retain_graph=False)[0]
            slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
            gradient_penalty = torch.mean((slopes - 1) ** 2)
            return gradient_penalty

        loss_discriminator_e = loss_discriminator_e + 0.5 * loss_GP(real_back_dat, fake_back_data, self.discriminator_E)
        loss_discriminator_s = loss_discriminator_s + 0.5 * loss_GP(real_fore_dat, fake_fore_data, self.discriminator_S)

        loss_discriminator_e = torch.log(torch.exp(loss_discriminator_e.mean()))
        loss_discriminator_s = torch.log(torch.exp(loss_discriminator_s.mean()))
        loss_Ge_Gs = torch.log(torch.exp(loss_Ge_Gs.mean()))

        # if torch.isnan(loss_Ge_Gs):
        #     print('out_discriminator_E_fake:',out_discriminator_E_fake)
        #     print('loss_discriminator_e:', loss_discriminator_e)
        #     print('out_discriminator_S_fake:', out_discriminator_S_fake)
        #     print('self.discriminator_S(real_fore_dat):', self.discriminator_S(real_fore_dat))
        #     print('loss_discriminator_s:', loss_discriminator_s)

        loss_dict = {'disc-loss-E': loss_discriminator_e, 'disc-loss-S': loss_discriminator_s,
                     'gene-loss-ES': loss_Ge_Gs
                     }

        loss = loss_dict
        eq_num, output = None, fake_back_data
        return loss, eq_num, label, output

    def training_mode(self, pretrain=False, datasetName='crwu', **kwargs):
        start_epoch = 0
        pretrain = False
        if pretrain:
            self.net, start_epoch = self.load_model_weight_auto(self.net, **kwargs)
        device = self.device
        net = self.net.to(device)

        optimizer_GE = self.optimizer_GE
        scheduler_GE = self.scheduler_GE
        optimizer_GS = self.optimizer_GS
        scheduler_GS = self.scheduler_GS
        optimizer_DE = self.optimizer_DE
        scheduler_DS = self.scheduler_DS
        optimizer_DS = self.optimizer_DS
        scheduler_DS = self.scheduler_DS

        """ training system initialization before start training (only one step in advance)"""
        self.training_init_save()
        print('[train]: ================ starting train ========================')
        """ start training """
        for idx_epo in range(start_epoch, self.epochNum):
            net.train()
            train_loss = 0
            loss_dicts = torch.zeros([3])
            out = None
            mini_batch_num = 0
            # with torch.autograd.set_detect_anomaly(True):
            for i, dat in enumerate(self.train_dataLoader):
                ''' Design program '''
                loss_dict, eq_num, label, output = self.format_train(dat, net, datasetName, device)

                loss_discriminator_e = loss_dict['disc-loss-E']
                loss_discriminator_s = loss_dict['disc-loss-S']
                loss_Ge_Gs = loss_dict['gene-loss-ES']
                # print('=============--------------------------', len(dat[0]), i)
                # if mini_batch_num >= self.config['batchSize']:

                loss_discriminator_e.backward(retain_graph=True)
                loss_discriminator_s.backward(retain_graph=True)
                loss_Ge_Gs.backward(retain_graph=False)
                # loss = loss_discriminator_e + loss_discriminator_s + loss_Ge_Gs
                # loss.backward(retain_graph=False)  #
                optimizer_GE.step()
                optimizer_GS.step()
                optimizer_DE.step()
                optimizer_DS.step()

                optimizer_GE.zero_grad()
                optimizer_GS.zero_grad()
                optimizer_DS.zero_grad()
                optimizer_DE.zero_grad()


                train_loss += float(loss_discriminator_e.item() + loss_discriminator_s.item() + loss_Ge_Gs.item())
                mini_batch_num += len(label)

                loss_dicts = loss_dicts + torch.tensor([float(loss_discriminator_e.item()),
                             float(loss_discriminator_s.item()),
                             float(loss_Ge_Gs.item())
                             ])
                out = output

            torch.cuda.empty_cache()
            train_loss = train_loss / len(self.train_dataLoader)
            ''' dynamic adjust lr '''
            # scheduler.step(train_loss)

            loss_dict = loss_dicts / len(self.train_dataLoader)
            loss_dict = {'disc-loss-E': loss_dict[0],
                         'disc-loss-S': loss_dict[1],
                         'gene-loss-ES': loss_dict[2]
                         }

            ''' record log '''  # result must the same as the log_dict(above)!
            result = {'loss': train_loss, **loss_dict}
            self.summary_log.add_scalars('train', result, idx_epo, tolerant=True)
            self.summary_log.plot_output_in_fig(out[:2], 'generate_e_'+str(idx_epo))
            ''' test model online '''
            if (idx_epo + 1) % 1 == 0:
                result = self.test_model(net, self.test_dataLoader, pretrain=False, datasetName=datasetName)
                self.summary_log.add_scalars('test', result, idx_epo, tolerant=True)

            ''' 保存模型 '''
            # if train_loss < self.summary_log.getMin('train', 'loss', rm_last=True):
                # self.saveModel_onlyOne(self.net, idx_epo, 'best')
            # if (idx_epo + 1) % 5 == 0:
            #     self.saveModel_onlyOne(self.net, idx_epo, name='')
            def save(net, name):
                filepath = os.path.join(self.resultPath, 'model', 'checkpoint_' + name + '.model')
                state = {'Net': net.state_dict()}
                torch.save(state, filepath)
                print('save models in', filepath)
                # print(f'')

            save(self.generate_E, 'generate_E')
            save(self.generate_S, 'generate_S')
            save(self.discriminator_E, 'discriminator_E')
            save(self.discriminator_S, 'discriminator_S')
            print('')

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
        loss_dicts = torch.zeros([3])
        train_loss = 0
        mini_batch_num = 0
        with torch.autograd.set_detect_anomaly(True):
            for i, dat in enumerate(dataLoader):
                ''' Design program '''
                loss_dict, eq_num, label, output = self.format_train(dat, net, datasetName, device)
                loss_discriminator_e = loss_dict['disc-loss-E']
                loss_discriminator_s = loss_dict['disc-loss-S']
                loss_Ge_Gs = loss_dict['gene-loss-ES']

                train_loss += float(loss_discriminator_e.item() + loss_discriminator_s.item() + loss_Ge_Gs.item())
                mini_batch_num += len(label)
                loss_dicts += torch.tensor([float(loss_discriminator_e.item()),
                                           float(loss_discriminator_s.item()),
                                           float(loss_Ge_Gs.item())
                                           ])

        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)

        loss_dict = loss_dicts / len(self.train_dataLoader)
        loss_dict = {'disc-loss-E': loss_dict[0],
                     'disc-loss-S': loss_dict[1],
                     'gene-loss-ES': loss_dict[2]
                     }
        result = {'loss': train_loss, **loss_dict}
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

        for i, dat in enumerate(dataLoader):
            ''' Design program '''
            loss, eq_num, label, output = self.format_train(dat, net, datasetName, device)

            acc_num += eq_num
            mini_batch_num += len(label)
            train_loss += float(loss.item())

            vs.add_data_series(
                data={'label': label.detach().cpu().numpy(), 'predict': output.argmax(dim=1).detach().cpu().numpy()})
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
