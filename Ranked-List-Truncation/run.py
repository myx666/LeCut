import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch as t
import torch.optim as optim
import logging
import configparser
import random
import shutil
import json

from dataloader import *
from models import *
from utils import losses
from utils.metrics import Metric

# from tensorboardX import SummaryWriter

RUNNING_PATH = '/work/mayixiao/cutoff/Ranked-List-Truncation'
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, args):
        """trainer for the truncation model
        
        Args:
            args: args for training
        """
        # params for training
        self.seq_len = 100#300 if args.retrieve_data == 'robust04' else 40
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.lr = args.lr
        self.cuda = args.cuda
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.istrain = args.istrain
        self.target = args.criterion
        self.iter_num = args.iter
        self.retrieve_data = args.retrieve_data
        self.dataset_name = args.dataset_name
        self.tail_name = args.model_path[-8:-4]

        self.best_test_f1 = -float('inf')
        self.best_test_dcg = -float('inf')
        self.best_test_nci = -float('inf')
        self.f1_record = []
        self.dcg_record = []
        self.nci_record = []

        if self.model_name == 'bicut': 
            # self.train_loader, self.test_loader = bc_dataloader(args.dataset_name, args.batch_size, args.num_workers)
            self.train_loader, self.test_loader, rank_data = base_dataloader(args.model_name, args.retrieve_data, args.dataset_name, args.batch_size)
            input_dim = len(rank_data.getX_train()[0][0])
            self.model = BiCut(input_size=input_dim, dropout=self.dropout)
            self.criterion = losses.BiCutLoss(metric=args.criterion)
        elif self.model_name == 'choppy':
            self.train_loader, self.test_loader, rank_data = base_dataloader(args.model_name, args.retrieve_data, args.dataset_name, args.batch_size)
            self.model = Choppy(seq_len=len(rank_data.getX_train()[0]), dropout=self.dropout)
            self.criterion = losses.ChoppyLoss(metric=args.criterion)
        elif self.model_name == 'attncut':
            self.train_loader, self.test_loader, rank_data = base_dataloader(args.model_name, args.retrieve_data, args.dataset_name, args.batch_size)
            input_dim = len(rank_data.getX_train()[0][0])
            self.model = AttnCut(input_size=input_dim, dropout=self.dropout)
            self.criterion = losses.AttnCutLoss(metric=args.criterion)
        elif self.model_name == 'lecut':
            self.train_loader, self.test_loader, rank_data = base_dataloader(args.model_name, args.retrieve_data, args.dataset_name, args.batch_size, self.iter_num)
            input_dim = len(rank_data.getX_train()[0][0])
            seq_len = len(rank_data.getX_train()[0])
            self.model = LeCut(seq_len=seq_len, input_size=input_dim, dropout=self.dropout)
            self.criterion = losses.LeCutLoss(metric=args.criterion)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)
        
        if self.cuda: 
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        if (not args.istrain) and os.path.exists(self.model_path): 
            self.load_model()
        # self.writer = SummaryWriter(log_dir=args.Tensorboard_dir)
            
    def train_epoch(self, epoch):
        """train stage for every epoch
        """
        start_time = time.time()
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        train_len = 0
        step, num_itr = 0, len(self.train_loader)
        logging.info('-' * 100)
        for X_train, y_train in tqdm(self.train_loader, desc='Training for epoch_{}'.format(epoch)):
            self.model.train()
            self.optimizer.zero_grad()
            if self.cuda: X_train, y_train = X_train.cuda(), y_train.cuda()

            output = self.model(X_train)
            loss = self.criterion(output, y_train)

            loss.backward()
            self.optimizer.step()
            
            if self.model_name == 'bicut':
                predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                k_s = []
                for results in predictions:
                    if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                    else: k_s.append(np.argmin(results)+1)
                # print(k_s)
            else: 
                predictions = output.detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1) + 1
            y_train = y_train.data.cpu().numpy()
            f1, _ = Metric.f1(y_train, k_s)
            dcg, _ = Metric.dcg(y_train, k_s)
            # self.writer.add_scalar('train/loss_step', loss.item(), step + num_itr * epoch)

            epoch_loss += loss.item()
            epoch_f1 += f1*len(y_train)
            epoch_dcg += dcg*len(y_train)
            step += 1
            train_len += len(y_train)

        train_loss, train_f1, train_dcg = epoch_loss / step, epoch_f1 / train_len, epoch_dcg / train_len
        # self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
        # self.writer.add_scalar('train/F1_epoch', train_f1, epoch)
        # self.writer.add_scalar('train/DCG_epoch', train_dcg, epoch)
        logging.info('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        logging.info('\tTrain: loss = {} | f1 = {:.6f} | dcg = {:.6f}\n'.format(train_loss, train_f1, train_dcg))

    def apply_dropout(self, m):
        if type(m) == t.nn.Dropout:
            m.train()

    def to_data_raw(self, lines, type=1): # from neural results to score dics
        if type == 1:
            data_raw = {}
            for line in lines:
                dic = eval(line)
                qid, cid = dic['id_'].split('_')
                if qid not in data_raw:
                    data_raw[qid] = {cid: dic['res'][1] - dic['res'][0]}
                else:# len(list(data_raw[qid].keys())) < 100:
                    data_raw[qid][cid] = dic['res'][1] - dic['res'][0]
        else:
            data_raw = lines
        sorted_data_raw = {}
        for key in data_raw:
            sorted_results = sorted(data_raw[key].items(), key=lambda x: x[1], reverse=True)
            sorted_data_raw[key] = {item[0]:item[1] for item in sorted_results}
        # print({key:sorted_data_raw[key].keys() for key in sorted_data_raw})
        return sorted_data_raw

    def test(self, epoch):
        """test stage for every epoch
        """
        
        epoch_loss, epoch_f1, epoch_dcg, epoch_nci = 0, 0, 0, 0
        test_len = 0
        step = 0
        
        if self.istrain:
            for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
                self.model.eval()
                if self.model_name == 'bicut':
                    self.model.apply(self.apply_dropout)
                with t.no_grad():
                    if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                    output = self.model(X_test)
                    loss = self.criterion(output, y_test)

                    
                    if self.model_name == 'bicut':
                        predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                        k_s = []
                        for results in predictions:
                            if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                            else: k_s.append(np.argmin(results) + 1)
                        # print(k_s)
                    else: 
                        predictions = output.detach().cpu().squeeze().numpy()
                        k_s = np.argmax(predictions, axis=1) + 1
                        # print(k_s)
                    y_test = y_test.data.cpu().numpy()
                    f1, _ = Metric.f1(y_test, k_s)
                    dcg, _ = Metric.dcg(y_test, k_s)
                    nci, _ = Metric.nci(y_test, k_s, self.retrieve_data)
                    
                    epoch_loss += loss.item()
                    epoch_f1 += f1*len(y_test)
                    epoch_dcg += dcg*len(y_test)
                    epoch_nci += nci*len(y_test)
                    step += 1
                    test_len += len(y_test)
        else: # output cut-off score
            sig_test_dic = {'f1':{}, 'dcg':{}, 'nci':{}} # for significance test
            w_score = {}
            scores_train = self.to_data_raw(json.load(open('/work/mayixiao/cutoff/results/l2r_%s_%s_train_%i.json'%(self.retrieve_data, self.dataset_name, self.iter_num),'r')),2)
            scores_test = self.to_data_raw(json.load(open('/work/mayixiao/cutoff/results/l2r_%s_%s_test_%i.json'%(self.retrieve_data, self.dataset_name, self.iter_num),'r')),2)
            all_keys = []
            for X_test, y_test in tqdm(self.train_loader, desc='Test after epoch_{}'.format(epoch)):
                self.model.eval()
                with t.no_grad():
                    X_test_list = X_test.numpy().tolist() # 20*100*770
                    if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                    output = self.model(X_test)
                    X_scores = [[ x[0] for x in x_list] for x_list in X_test_list]
                    for i, X_score in enumerate(X_scores):
                        for key in scores_train:
                            if [round(a,2) for a in list(scores_train[key].values())]  == [round(a,2) for a in X_score]: 
                                # print(i, key)
                                all_keys.append(key)
                                
                                output_list = output.cpu().detach().numpy().tolist()[i]
                                w_score[key] = {key2: score[0] for key2, score in zip(list(scores_train[key].keys()), output_list)}
                                
                    # w_score.extend()
                    
            for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
                sig_epoch = [] 
                self.model.eval()
                if self.model_name == 'bicut':
                    self.model.apply(self.apply_dropout)
                with t.no_grad():
                    X_test_list = X_test.numpy().tolist() # 20*100*770
                    if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                    output = self.model(X_test)
                    loss = self.criterion(output, y_test)

                    
                    X_scores = [[ x[0] for x in x_list] for x_list in X_test_list]
                    for i, X_score in enumerate(X_scores):
                        for key in scores_test:
                            if [round(a,3) for a in list(scores_test[key].values())]  == [round(a,3) for a in X_score]: 
                                # print(key)
                                all_keys.append(key)
                                sig_epoch.append(key)
                                output_list = output.cpu().detach().numpy().tolist()[i]
                                w_score[key] = {key2: score[0] for key2, score in zip(list(scores_test[key].keys()), output_list)}
                    
                    if self.model_name == 'bicut':
                        predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                        k_s = []
                        for results in predictions:
                            if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                            else: k_s.append(np.argmin(results) + 1)
                    else: 
                        predictions = output.detach().cpu().squeeze().numpy()
                        k_s = np.argmax(predictions, axis=1) + 1
                        
                    y_test = y_test.data.cpu().numpy()
                    f1, f1_epoch = Metric.f1(y_test, k_s)
                    dcg, dcg_epoch = Metric.dcg(y_test, k_s)
                    nci, nci_epoch = Metric.nci(y_test, k_s, self.retrieve_data)
                    epoch_loss += loss.item()
                    epoch_f1 += f1*len(y_test)
                    epoch_dcg += dcg*len(y_test)
                    epoch_nci += nci*len(y_test)
                    step += 1
                    test_len += len(y_test)
                    for i in range(len(sig_epoch)):
                        sig_test_dic['f1'][sig_epoch[i]] = f1_epoch[i]
                        sig_test_dic['dcg'][sig_epoch[i]] = dcg_epoch[i]
                        sig_test_dic['nci'][sig_epoch[i]] = nci_epoch[i]

            print(len(set(all_keys)), test_len)
            json.dump(w_score, open('/work/mayixiao/cutoff/results/%s.json'%(self.model_name + '_' + self.retrieve_data + '_' + self.dataset_name + '_' + str(self.iter_num)),'w'), ensure_ascii=False)

            # if self.iter_num == 1:
                # json.dump(sig_test_dic, open('/work/mayixiao/cutoff/results/sig/%s.json'%(self.model_name + '_' + self.retrieve_data + '_' + self.dataset_name + '_' + self.tail_name),'w'), ensure_ascii=False)
        
        test_loss, test_f1, test_dcg, test_nci = epoch_loss / step, epoch_f1 / test_len, epoch_dcg / test_len, epoch_nci / test_len
        # self.writer.add_scalar('test/loss_epoch', test_loss, epoch)
        # self.writer.add_scalar('test/F1_epoch', test_f1, epoch)
        # self.writer.add_scalar('test/DCG_epoch', test_dcg, epoch)
        self.f1_record.append(test_f1)
        self.dcg_record.append(test_dcg)
        self.nci_record.append(test_nci)
        logging.info('\tTest: loss = {} | f1 = {:.6f} | dcg = {:.6f} | nci = {:.6f}\n'.format(test_loss, test_f1, test_dcg, test_nci))
        
        if test_f1 > self.best_test_f1:
            self.best_test_f1 = test_f1
            if self.istrain and self.target == 'f1': 
                self.save_model()
        if test_dcg > self.best_test_dcg: 
            self.best_test_dcg = test_dcg
            if self.istrain and self.target == 'dcg': 
                self.save_model()
        if test_nci > self.best_test_nci: 
            self.best_test_nci = test_nci
            if self.istrain and self.target == 'nci': 
                self.save_model()

    def save_model(self):
        """save the best model
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        t.save(self.model.state_dict(), self.save_path + '{}.pkl'.format(self.model_name+ '_' + self.retrieve_data + '_' + self.dataset_name + '_' + str(self.iter_num)))
        logging.info('The best model has beed updated and saved in {}\n'.format(self.save_path))

    def load_model(self):
        """load the saved model
        """
        self.model.load_state_dict(t.load(self.model_path))
        logging.info('The best model has beed loaded from {}\n'.format(self.model_path))

    def run(self):
        """run the model
        """
        if self.istrain:
            logging.info('\nTrain the {} model: \n'.format(self.model_name))
            for epoch in range(self.epochs):
                self.train_epoch(epoch)
                self.test(epoch)
            best5_f1 = sum(sorted(self.f1_record, reverse=True)[:5]) / 5
            best5_dcg = sum(sorted(self.dcg_record, reverse=True)[:5]) / 5
            best5_nci = sum(sorted(self.nci_record, reverse=True)[:5]) / 5
            logging.info('the best metric of this model: f1: {} | dcg: {} | nci: {}'.format(self.best_test_f1, self.best_test_dcg, self.best_test_nci))
            logging.info('the best-5 metric of this model: f1: {} | dcg: {} | nci: {}'.format(best5_f1, best5_dcg, best5_nci))
        else:
            logging.info('\nTest the {} model: \n'.format(self.model_name))
            self.test(1)
            logging.info('the best metric of this model: f1: {} | dcg: {} | nci: {}'.format(self.best_test_f1, self.best_test_dcg, self.best_test_nci))
        


def main():
    """训练过程的主函数，用于接收训练参数等
    """
    parser = argparse.ArgumentParser(description="Truncation Model Trainer Args")
    parser.add_argument('--retrieve-data', type=str, default='LeCaRD')
    parser.add_argument('--dataset-name', type=str, default='slr')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='attncut')
    parser.add_argument('--criterion', type=str, default='f1')
    parser.add_argument('--model-path', type=str, default=None)
    # parser.add_argument('--ft', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='{}/best_model/'.format(RUNNING_PATH))
    parser.add_argument('--epochs', type=int, default=80) #80
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--istrain', type=int, default=1)
    parser.add_argument('--iter', type=int, default=1)

    args = parser.parse_args()
    args.cuda = t.cuda.is_available()
    args.Tensorboard_dir = '{}/Tensorboard_summary/Truncation'.format(RUNNING_PATH)
    if args.model_path == None:
        args.model_path = args.save_path + '{}.pkl'.format(args.model_name)
    
    if os.path.exists(args.Tensorboard_dir): shutil.rmtree(args.Tensorboard_dir)
    os.makedirs(args.Tensorboard_dir)
    
    config = configparser.ConfigParser()
    config.read('{}/hyper_parameter_{}.conf'.format(RUNNING_PATH, args.dataset_name))
    args.lr = config.getfloat('{}_conf'.format(args.model_name), 'lr')
    if args.retrieve_data == 'robust04': args.batch_size = config.getint('{}_conf'.format(args.model_name), 'batch_size')
    args.dropout = config.getfloat('{}_conf'.format(args.model_name), 'dropout')
    args.weight_decay = config.getfloat('{}_conf'.format(args.model_name), 'weight_decay')
   
    logging.info('{}'.format(vars(args)))
    trainer = Trainer(args)
    trainer.run()
    # trainer.writer.close()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
    main()
