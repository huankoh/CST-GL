import torch.optim as optim
from stgnn import *
import util
class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, scaling_required=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.loss = util.masked_mse
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.scaling_required = scaling_required

    def train(self, input, real_val, idx=None):
        '''
        Model training on train data.
        :param input: sliding window of time series observation
        :param real_val: Each observation at time t
        :return: mse loss, mape and rmse
        '''
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        if self.scaling_required:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = output
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1

        loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        self.iter += 1
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        '''
        Model evaluation using validation set.
        :param input: sliding window of time series observation
        :param real_val: Each observation at time t
        :return: mse loss, mape and rmse
        '''
        self.model.eval()
        output, _ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        if self.scaling_required:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = output

        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(), mape, rmse

    def pred(self, input):
        '''
        Model inference on test data (anomaly detection task).
        :param input: sliding window of time series observation
        :return: output - one step forecast, adp - learned uni-directed graph
        '''
        self.model.eval()
        output, adp = self.model(input)
        return output, adp
