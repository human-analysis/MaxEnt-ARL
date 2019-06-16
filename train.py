# train.py

import os
import torch
import numpy as np
import torch.nn as nn
from loss import EntropyLoss
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.util import AverageMeter, accuracy, Logger


class MaxentNet:
    def __init__(self, data,
                 train_loader=None,
                 test_loader=None,
                 total_epoch=200,
                 alpha=0.1,
                 epsilon=0.1,
                 use_cuda=False,
                 resume=False,
                 ckpt_filename=None,
                 resume_filename=None,
                 privacy_flag=True,
                 privacy_option='maxent-arl',
                 print_interval_train=10,
                 print_interval_test=10
                 ):
        # data info
        self.data = data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_sensitive_class = self.data.n_sensitive_class
        self.n_target_class = self.data.n_target_class

        # models
        self.adv_net = data.adversary_net
        self.target_net = data.target_net
        self.discriminator_net = data.discriminator_net

        # optimizer
        self.optimizer = data.optimizer
        self.discriminator_optimizer = data.discriminator_optimizer
        self.adv_optimizer = data.adv_optimizer
        self.target_optimizer = data.target_optimizer

        # loss
        self.kl_loss = nn.KLDivLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.entropy_loss = EntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.mse_loss = nn.MSELoss()

        # filename
        self.log_file_name = ckpt_filename+"_log.txt"
        self.adv_log_file_name = ckpt_filename+"_adv_log.txt"
        self.target_log_file_name = ckpt_filename + "_target_log.txt"
        self.checkpoint_filename = ckpt_filename
        self.adv_checkpoint_filename = ckpt_filename+"_adv.ckpt"
        self.target_checkpoint_filename = ckpt_filename + "_target.ckpt"

        # algorithm and visualization parameters
        self.alpha = torch.tensor([alpha*1.0], requires_grad=True)
        self.resume = resume
        self.epoch = 0
        self.gamma_param = 0.01
        self.plot_interval = 10
        self.print_interval_train = print_interval_train
        self.print_interval_test = print_interval_test
        self.use_cuda = use_cuda
        self.privacy_flag = privacy_flag
        self.privacy_option = privacy_option

        # local variables
        self.uniform = torch.tensor(1 / (self.data.n_sensitive_class)).repeat(self.data.n_sensitive_class)
        self.target_label = torch.zeros(0, dtype=torch.long)
        self.sensitive_label = torch.zeros(0, dtype=torch.long)
        self.sensitive_label_onehot = torch.FloatTensor(0, self.data.n_sensitive_class)
        self.target_label_onehot = torch.FloatTensor(0, self.data.n_target_class)
        self.inputs = torch.zeros(0, 0, 0)
        self.inputs.requires_grad = False
        self.batch_uniform = torch.FloatTensor(0, self.data.n_sensitive_class)
        self.epsilon = torch.tensor([epsilon]).float()

        if resume:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            if self.use_cuda:
                checkpoint = torch.load(os.path.join('checkpoint/', resume_filename))
            else:
                checkpoint = torch.load(os.path.join('checkpoint/',resume_filename), map_location=lambda storage, loc: storage)
            self.net = checkpoint['net']
            self.best_acc = 0  # checkpoint['acc']
            self.start_epoch = 0  # checkpoint['epoch']
            self.total_epoch = total_epoch  # + self.start_epoch

            for param in self.net.parameters():
                param.requires_grad = True
        else:
            self.net = data.net
            self.best_acc = 0
            self.start_epoch = 0
            self.total_epoch = total_epoch

        if self.use_cuda:
            self.net = self.net.cuda()
            self.discriminator_net = self.discriminator_net.cuda()
            self.adv_net = self.adv_net.cuda()
            self.target_net = self.target_net.cuda()
            self.net = nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            self.target_net = nn.DataParallel(self.target_net, device_ids=range(torch.cuda.device_count()))
            self.discriminator_net = nn.DataParallel(self.discriminator_net, device_ids=range(torch.cuda.device_count()))
            self.adv_net = nn.DataParallel(self.adv_net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            self.inputs = self.inputs.cuda()
            self.target_label = self.target_label.cuda()
            self.sensitive_label = self.sensitive_label.cuda()
            self.sensitive_label_onehot = self.sensitive_label_onehot.cuda()
            self.target_label_onehot = self.target_label_onehot.cuda()
            self.uniform = self.uniform.cuda()
            self.batch_uniform = self.batch_uniform.cuda()
            self.alpha = self.alpha.cuda()

        self.best_loss = 1e16
        self.adv_best_acc = 0
        self.target_best_acc = 0
        self.t_losses, self.t_top1, self.d_losses, self.d_top1 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        self.e_losses, self.losses = AverageMeter(), AverageMeter()
        self.t_top5, self.d_top5 = AverageMeter(), AverageMeter()
        self.adv_losses, self.adv_top1, self.adv_top5, self.entropy_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        self.target_losses, self.target_top1, self.target_top5, self.target_entropy_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    def perform_epoch(self, epoch, test_flag=False):
        if test_flag:
            self.net.eval()
            self.discriminator_net.eval()
            self.target_net.eval()
            loader = self.test_loader
            string = "Test"
            print_interval = self.print_interval_test
            data_size = len(self.test_loader)
        else:
            self.net.train()
            self.discriminator_net.train()
            self.target_net.train()
            loader = self.train_loader
            string = "Train"
            print_interval = self.print_interval_train
            data_size = len(self.train_loader)

        iteration = 0

        self.t_losses.reset()
        self.e_losses.reset()
        self.losses.reset()
        self.d_losses.reset()
        self.t_top1.reset()
        self.d_top1.reset()
        self.t_top5.reset()
        self.d_top5.reset()
        self.entropy_losses.reset()

        for batch_idx, (inputs, target_label, sensitive_label) in enumerate(loader):

            batch_size = inputs.size(0)
            iteration += 1

            self.inputs.resize_(inputs.size()).copy_(inputs)
            self.target_label.resize_(target_label.size()).copy_(target_label)
            self.sensitive_label.resize_(sensitive_label.size()).copy_(sensitive_label)
            self.sensitive_label_onehot.resize_([batch_size, self.data.n_sensitive_class])
            self.sensitive_label_onehot.zero_()
            self.sensitive_label_onehot.scatter_(1, torch.unsqueeze(self.sensitive_label, 1), 1)
            self.target_label_onehot.resize_([batch_size, self.data.n_target_class])
            self.target_label_onehot.zero_()
            self.target_label_onehot.scatter_(1, torch.unsqueeze(self.target_label, 1), 1)
            self.batch_uniform.resize_([batch_size, self.data.n_sensitive_class])
            self.batch_uniform[:, :] = 1.0/(self.data.n_sensitive_class)
            self.batch_uniform.scatter_(1, torch.unsqueeze(self.sensitive_label, 1), 0)
            self.optimizer.zero_grad()

            _, z, e_prob = self.net(self.inputs)
            target_outputs, _, t_prob = self.target_net(z)
            t_loss = self.nll_loss(torch.log(t_prob+1e-16), self.target_label)
            entropy_loss = torch.tensor(0)
            s_loss = torch.tensor(0)

            if self.privacy_flag:
                d_outputs, _, d_prob = self.discriminator_net(z)
                entropy_loss = -self.entropy_loss(d_prob)

                if self.privacy_option is 'maxent-arl':
                    s_loss = -entropy_loss  # self.kl_loss(torch.log(self.uniform.repeat(batch_size, 1)), d_prob)
                if self.privacy_option is 'ml-arl':
                    s_loss = -self.nll_loss(torch.log(d_prob+1e-16), self.sensitive_label)

                loss = (1-self.alpha)*t_loss + self.alpha*s_loss
            else:
                loss = t_loss

            if not test_flag:  # update weights
                self.optimizer.zero_grad()
                self.target_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.target_optimizer.step()

            # measure accuracy and record loss for learner
            t_prec1 = accuracy(t_prob.data, self.target_label.data)
            t_prec5 = accuracy(t_prob.data, self.target_label.data, topk=(int(np.min([5, self.n_target_class])),))
            self.t_losses.update(t_loss.data.item(), batch_size)
            self.e_losses.update(s_loss.data.item(), batch_size)
            self.losses.update(loss.data.item(), batch_size)
            self.t_top1.update(t_prec1[0], batch_size)
            self.t_top5.update(t_prec5[0], batch_size)
            self.entropy_losses.update(entropy_loss.data.item(), batch_size)

            if self.privacy_flag:
                if not test_flag:
                    self.discriminator_net.train()
                d_outputs, _, a_prob = self.discriminator_net(z.detach())
                d_loss = self.nll_loss(torch.log(a_prob+1e-16), self.sensitive_label)

                if not test_flag:
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                d_prec1 = accuracy(a_prob.data, self.sensitive_label.data)
                d_prec5 = accuracy(a_prob.data, self.sensitive_label.data, topk=(int(np.min([5, self.n_sensitive_class])),))
                self.d_losses.update(d_loss.data.item(), batch_size)
                self.d_top1.update(d_prec1[0], batch_size)
                self.d_top5.update(d_prec5[0], batch_size)

                if iteration % print_interval == 0:
                    print(string + '_Epoch:[{0}][{1}/{2}] |'
                          ' T_Loss: {3:.2f} |'
                          ' E_Loss: {4:.2f} |'
                          ' Loss: {5:.2f} |'
                          ' T_Prec: {6:.2f} |'
                          ' T_Prec5: {7:.2f} |'
                          ' D_Loss: {8:.2f} |'
                          ' D_Prec: {9:.2f} |'
                          ' D_Prec5: {10:.2f} |'
                          ' D_Entropy: {11:.2f} |'
                        .format(
                        epoch, batch_idx, data_size,
                        float(self.t_losses.avg), float(self.e_losses.avg),
                        float(self.losses.avg),float(self.t_top1.avg.item()),
                        float(self.t_top5.avg.item()), float(self.d_losses.avg),
                        float(self.d_top1.avg.item()), float(self.d_top5.avg.item()),
                        float(self.entropy_losses.avg)))

            else:
                if iteration % print_interval == 0:
                    print(string + '_Epoch:[{0}][{1}/{2}] |'
                          ' T_Loss: {3:.2f} |'
                          ' T_Prec: {4:.2f} |'
                          ' T_Prec5: {5:.2f} |'
                        .format(
                        epoch, batch_idx, data_size,
                        float(self.t_losses.avg), float(self.t_top1.avg.item()),
                        float(self.t_top5.avg.item())))

        return self.losses.avg, self.t_top1.avg, self.t_top5.avg, self.d_losses.avg, self.d_top1.avg, self.d_top5.avg, self.entropy_losses.avg

    def train(self):
        self.logger = Logger(os.path.join('checkpoint/', self.log_file_name), title='Problem')
        self.logger.set_names(['LR', 'Train-Loss', 'Test-Loss', 'Train-Acc.', 'Train-Acc5.', 'Test Acc.', 'Test Acc5.',
                               'D-Train Loss', 'D-Test Loss', 'D-Train Acc.', 'D-Train Acc5.', 'D-Test Acc.', 'D-Test Acc5.', 'D-Train-Entropy',
                               'D-Test-Entropy'])

        scheduler1 = CosineAnnealingLR(self.optimizer, T_max=self.total_epoch, eta_min=1e-7)
        scheduler2 = CosineAnnealingLR(self.discriminator_optimizer, T_max=self.total_epoch, eta_min=1e-6)

        for epoch in range(self.start_epoch, self.total_epoch):
            print('\nEpoch: %d' % epoch)
            scheduler1.step()
            scheduler2.step()

            train_loss, train_acc, train_acc5, d_train_loss, d_train_acc, d_train_acc5, d_train_entropy = self.perform_epoch(epoch=epoch, test_flag=False)

            with torch.no_grad():
                test_loss, test_acc, test_acc5, d_test_loss, d_test_acc, d_test_acc5, d_test_entropy = self.perform_epoch(epoch=epoch, test_flag=True)

            self.logger.append([self.optimizer.param_groups[0]['lr'], float(train_loss), float(test_loss), float(train_acc), float(train_acc5),
                           float(test_acc), float(test_acc5), float(d_train_loss), float(d_test_loss), float(d_train_acc), float(d_train_acc5),
                           float(d_test_acc), float(d_test_acc5), float(d_train_entropy), float(d_test_entropy)])

            # it is optimum only when we reach the end of the game by optimization,
            # any other value e.g. current discriminator feedback is non-optimal
            if (epoch + 1) % 10:
                print('Saving..')  # Save checkpoint.
                state = {
                    'net': self.net.module if self.use_cuda else self.net,
                    'state_dict': self.net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, 'checkpoint/' + self.checkpoint_filename + '.ckpt')
                self.best_acc = test_acc
                self.best_loss = test_loss

        self.logger.close()
        print("Done")

    def perform_epoch_adversary(self, epoch, test_flag=False):
        if test_flag:
            self.adv_net.eval()
            loader = self.test_loader
            str = "Test"
            print_interval = self.print_interval_test
        else:
            self.adv_net.train()
            loader = self.train_loader
            str = "Train"
            print_interval = self.print_interval_train

        self.net.eval()
        iteration = 0
        self.adv_losses.reset()
        self.adv_top1.reset()
        self.adv_top5.reset()
        self.entropy_losses.reset()

        for batch_idx, (inputs, target_label, sensitive_label) in enumerate(loader):
            batch_size = inputs.size(0)
            iteration += 1
            if self.data.name == 'mnist':
                inputs = torch.unsqueeze(inputs, 1).float()

            self.inputs.resize_(inputs.size()).copy_(inputs)
            self.target_label.resize_(target_label.size()).copy_(target_label)
            self.sensitive_label.resize_(sensitive_label.size()).copy_(sensitive_label)

            with torch.no_grad():
                outputs, z, _ = self.net(self.inputs)

            d_outputs, _, prob = self.adv_net(z.detach())
            d_loss = self.cross_entropy_loss(d_outputs, self.sensitive_label)

            with torch.no_grad():
                entropy_loss = -self.entropy_loss(prob)

            if not test_flag:
                self.adv_optimizer.zero_grad()
                d_loss.backward()
                self.adv_optimizer.step()

            d_prec1 = accuracy(prob.data, self.sensitive_label.data)
            d_prec5 = accuracy(prob.data, self.sensitive_label.data, topk=(int(np.min([5, self.n_sensitive_class])),))
            self.adv_losses.update(d_loss.data.item(), batch_size)
            self.adv_top1.update(d_prec1[0], batch_size)
            self.adv_top5.update(d_prec5[0], batch_size)
            self.entropy_losses.update(entropy_loss.data.item(), batch_size)

            if iteration % print_interval == 0:
                print(str + ' Epoch:[{0}][{1}/{2}] |'
                      ' T_Loss: {3:.5f} |'
                      ' T_Prec: {4:.2f} |'
                      ' T5_Prec: {5:.2f} |'
                      ' Entropy: {6:.3f} |'
                    .format(
                    epoch, batch_idx, len(self.train_loader),
                    float(self.adv_losses.avg), float(self.adv_top1.avg.item()), float(self.adv_top5.avg),
                    float(self.entropy_losses.avg)))

        return self.adv_losses.avg, self.adv_top1.avg, self.adv_top5.avg, self.entropy_losses.avg

    def train_adversary(self, model_filename=None, total_epoch=100):
        self.adv_logger = Logger(os.path.join('checkpoint/', self.adv_log_file_name), title='Problem')
        self.adv_logger.set_names(['LR', 'Train-Loss', 'Test-Loss', 'Train Acc.', 'Train Acc5.', 'Test Acc.', 'Test Acc5.',
                                   'Train Entropy','Test Entropy'])

        self.adv_best_acc = 0
        scheduler = CosineAnnealingLR(self.adv_optimizer, T_max=total_epoch, eta_min=1e-7)
        if model_filename is not None:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join('checkpoint/', model_filename))
            self.net = checkpoint['net']
            self.net.eval()

        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            scheduler.step()
            train_loss, train_acc, train_acc5, train_entropy = self.perform_epoch_adversary(epoch=epoch, test_flag=False)
            with torch.no_grad():
                test_loss, test_acc, test_acc5, test_entropy = self.perform_epoch_adversary(epoch=epoch, test_flag=True)

            self.adv_logger.append([self.adv_optimizer.param_groups[0]['lr'], float(train_loss), float(test_loss), float(train_acc),
                           float(train_acc5), float(test_acc), float(test_acc5), float(train_entropy), float(test_entropy)])
            # Save checkpoint.
            if test_acc > self.adv_best_acc:
                print('Saving..')
                state = {
                    'net': self.adv_net.module if self.use_cuda else self.adv_net,
                    'state_dict': self.adv_net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    'optimizer': self.adv_optimizer.state_dict()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, 'checkpoint/' + self.adv_checkpoint_filename)
                self.adv_best_acc = test_acc

        self.adv_logger.close()
        print("Adversary Done.")

    def perform_epoch_target(self, epoch, test_flag=False):
        if test_flag:
            self.target_net.eval()
            loader = self.test_loader
            str = "Test"
            print_interval = self.print_interval_test
        else:
            self.target_net.train()
            loader = self.train_loader
            str = "Train"
            print_interval = self.print_interval_train

        self.net.eval()
        iteration = 0
        self.target_losses.reset()
        self.target_top1.reset()
        self.target_top5.reset()
        self.target_entropy_losses.reset()

        for batch_idx, (inputs, target_label, sensitive_label) in enumerate(loader):
            batch_size = inputs.size(0)
            iteration += 1
            if self.data.name == 'mnist':
                inputs = torch.unsqueeze(inputs, 1).float()

            self.inputs.resize_(inputs.size()).copy_(inputs)
            self.target_label.resize_(target_label.size()).copy_(target_label)
            self.sensitive_label.resize_(sensitive_label.size()).copy_(sensitive_label)

            with torch.no_grad():
                outputs, z, _ = self.net(self.inputs)

            d_outputs, _, prob = self.target_net(z.detach())
            d_loss = self.cross_entropy_loss(d_outputs, self.target_label)

            with torch.no_grad():
                entropy_loss = -self.entropy_loss(prob)

            if not test_flag:
                self.target_optimizer.zero_grad()
                d_loss.backward()
                self.target_optimizer.step()

            d_prec1 = accuracy(prob.data, self.target_label.data)
            d_prec5 = accuracy(prob.data, self.target_label.data, topk=(int(np.min([5, self.n_target_class])),))
            self.target_losses.update(d_loss.data.item(), batch_size)
            self.target_top1.update(d_prec1[0], batch_size)
            self.target_top5.update(d_prec5[0], batch_size)
            self.target_entropy_losses.update(entropy_loss.data.item(), batch_size)

            if iteration % print_interval == 0:
                print(str + ' Epoch:[{0}][{1}/{2}] |'
                      ' T_Loss: {3:.5f} |'
                      ' T_Prec: {4:.2} |'
                      ' T5_Prec: {5:.2f} |'
                      ' Entropy: {6:.3f} |'
                    .format(
                    epoch, batch_idx, len(loader),
                    float(self.target_losses.avg), float(self.target_top1.avg.item()), float(self.target_top5.avg),
                    float(self.target_entropy_losses.avg)))

        return self.target_losses.avg, self.target_top1.avg, self.target_top5.avg, self.target_entropy_losses.avg

    def train_target(self, model_filename=None, total_epoch=100):
        self.target_logger = Logger(os.path.join('checkpoint/', self.target_log_file_name), title='Problem')
        self.target_logger.set_names(['LR', 'Train-Loss', 'Test-Loss', 'Train Acc.', 'Train Acc5.', 'Test Acc.', 'Test Acc5.',
                                   'Train Entropy','Test Entropy'])

        self.target_best_acc = 0
        scheduler = CosineAnnealingLR(self.target_optimizer, T_max=total_epoch, eta_min=1e-7)
        if model_filename is not None:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join('checkpoint/', model_filename))
            self.net = checkpoint['net']
            self.net.eval()

        for epoch in range(total_epoch):
            print('\nEpoch: %d' % epoch)
            scheduler.step()
            train_loss, train_acc, train_acc5, train_entropy = self.perform_epoch_target(epoch=epoch, test_flag=False)
            with torch.no_grad():
                test_loss, test_acc, test_acc5, test_entropy = self.perform_epoch_target(epoch=epoch, test_flag=True)

            self.target_logger.append([self.target_optimizer.param_groups[0]['lr'], float(train_loss), float(test_loss), float(train_acc),
                           float(train_acc5), float(test_acc), float(test_acc5), float(train_entropy), float(test_entropy)])

            if test_acc > self.target_best_acc:
                print('Saving..')  # Save checkpoint.
                state = {
                    'net': self.target_net.module if self.use_cuda else self.target_net,
                    'state_dict': self.target_net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    'optimizer': self.target_optimizer.state_dict()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, 'checkpoint/' + self.target_checkpoint_filename)
                self.target_best_acc = test_acc

        self.target_logger.close()
        print("Target Done")