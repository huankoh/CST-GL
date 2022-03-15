import argparse
import time
from util import *
from trainer import Trainer
from stgnn import stgnn
from anomaly_dd import anomaly_dd
from evaluate import pointwise_evaluation, early_detection_evaluation
import json
import pandas as pd


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

# Data and Pre-processing
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='./data/minmax_swat', help='data path')
parser.add_argument('--scaling_required', type=bool, default=False, help='Whether to scale input for model and inverse scale output from model.')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=str, default='', help='experiment id')
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--save_result',type=str,default='',help='path to save forecasting results')

# For evaluation of early detection ability
parser.add_argument('--delays',type=str,default=[0,6,30,60,120,180,360],help='Early detection delay constraint values') # for wadi/swat every 6 timestamp is a minute

# Training and optimization
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')
parser.add_argument('--epochs', type=int, default=20, help='')
parser.add_argument('--print_every', type=int, default=5000, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

## CST-GNN Framework hyper-parameters
# MTCL layer
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--propalpha', type=float, default=0.1, help='prop alpha in graph module')
parser.add_argument('--tanhalpha', type=float, default=20, help='adj alpha in graph constructor')
parser.add_argument('--num_split', type=int, default=3, help='number of splits for graphs')
parser.add_argument('--node_dim', type=int, default=256, help='dim of nodes')
parser.add_argument('--num_nodes', type=int, default=51, help='number of nodes/variables')
parser.add_argument('--subgraph_size', type=int, default=15, help='k')

# STGNN layer
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')

parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')

parser.add_argument('--layers', type=int, default=2, help='number of layers')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=5, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')   # 1 if one-step forecast

# Graph-based Anomaly Detection
parser.add_argument('--normalization_window',type=int,default=None,help='Window size to normalize forecast error.')
parser.add_argument('--pca_compo',type=int,default=10,help='Number of principal components, L')
parser.add_argument('--error_batch_size',type=int,default=128,help='Batch processing sliding window normalization')

args = parser.parse_args()
torch.set_num_threads(4)
args.delays = list(map(int, args.delays.strip('[]').split(',')))

def main(runid):
    np.random.seed(runid)
    torch.manual_seed(runid)
    torch.cuda.manual_seed(runid)
    torch.cuda.manual_seed_all(runid)
    # random.seed(runid)
    os.environ['PYTHONHASHSEED'] = str(runid)

    # load data
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,args.scaling_required)
    scaler = dataloader['scaler']

    predefined_A = None

    model = stgnn(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    
    print(args)
    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.scaling_required)

    print("start training...", flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:,0,:,:],id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss


    ###############  Training completed and start forecasting  ###############
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    ##### train data #####
    outputs = []
    realy = torch.Tensor(dataloader['y_train']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds,_ = engine.pred(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds)

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    if args.scaling_required:
        pred = scaler.inverse_transform(yhat)
    else:
        pred = yhat

    train_pred = pred.squeeze().cpu().detach().numpy()
    train_label = realy.squeeze().cpu().detach().numpy()

    ##### val data #####
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, adp = engine.pred(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds)
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    if args.scaling_required:
        pred = scaler.inverse_transform(yhat)
    else:
        pred = yhat

    val_pred = pred.squeeze().cpu().detach().numpy()
    val_label = realy.squeeze().cpu().detach().numpy()

    ##### test data #####
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds, adp = engine.pred(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds)
    adp = adp.cpu().detach().numpy() # save a copy of learned pairwise correlation graph

    yhat = torch.cat(outputs, dim=0)  # WADI: (17408, 1, nodes, 1)
    yhat = yhat[:realy.size(0), ...]  # WADI: (17275, 1, nodes, 1)

    if args.scaling_required:
        pred = scaler.inverse_transform(yhat)
    else:
        pred = yhat

    test_pred = pred.squeeze().cpu().detach().numpy()
    test_label = realy.squeeze().cpu().detach().numpy()

    if args.save_result:
        # train
        np.save(args.save_results + 'train_pred_' + str(runid) + '.npy', train_pred)
        np.save(args.save_results + 'train_label_' + str(runid) + '.npy', train_label)
        # val
        np.save(args.save_results+'val_pred_'+str(runid) +'.npy',val_pred)
        np.save(args.save_results+'val_label_'+str(runid) +'.npy',val_label)
        # test
        np.save(args.save_results+'test_pred_'+str(runid) +'.npy',test_pred)
        np.save(args.save_results+'test_label_'+str(runid) +'.npy',test_label)
        # ADP - MTCL layer uni-directed graph
        np.save(args.save_results + "ADP_" + str(runid), adp)


    ############### Anomaly Detection and Diagnosis ###############
    anomaly_detector = anomaly_dd(train_label,val_label,test_label,train_pred,val_pred,test_pred,
                           args.normalization_window, args.error_batch_size)

    indicator, prediction = anomaly_detector.scorer(args.pca_compo)
    # Evaluate results
    with open(args.data+'/anomaly_labels.txt','r') as f:
        labels = [int(i) for i in f.read().split(',')]

    pointwise = pointwise_evaluation(labels,prediction,indicator)
    early = early_detection_evaluation(labels,indicator,args.delays)

    return pointwise, early


if __name__ == "__main__":


    overall = []
    early_detect = []

    for i in range(args.runs):
        pointwise, early = main(i)
        overall.append(pointwise)
        early_detect.append(early)

    df = pd.DataFrame(overall)
    mean = dict(df.mean().round(4))
    std = dict(df.std().round(4))


    print('\n\n-----------Overall Detection Results-----------\n\n')
    print('---- AUC result ----')
    table_data = [['Metric:','ROC-AUC','PRC-AUC'],
    ['mean:',mean['roc'],mean['prc']],
    ['std:',std['roc'],std['prc']]]
    for row in table_data:
        print("{: >20} {: >20} {: >20}".format(*row))

    print('---- Best F1 result ----')
    table_data = [['Metric:','Precision','Recall','F1'],
    ['mean:',mean['best_precision'],mean['best_recall'],mean['best_f1']],
    ['std:',std['best_precision'],std['best_recall'],std['best_f1']]]
    for row in table_data:
        print("{: >20} {: >20} {: >20} {: >20}".format(*row))

    print('---- Automatic threshold ----')
    table_data= [['Metric:','Precision','Recall','F1'],
    ['mean:',mean['precision'],mean['recall'],mean['f1']],
    ['std:',std['precision'],std['recall'],std['f1']]]
    for row in table_data:
        print("{: >20} {: >20} {: >20} {: >20}".format(*row))
    
    print('\n\n-----------Early Detection Results-----------\n\n')
    df = pd.DataFrame(early_detect)
    mean = dict(df.mean().round(4))
    std = dict(df.std().round(4))
    table_data= [['Delay']+[str(d) for d in args.delays],
    ['mean:'] + [str(mean['delay_'+str(d)]) for d in args.delays],
    ['std:']+[str(std['delay_'+str(d)]) for d in args.delays]]
    
    print_holder = "{: >20} "*(len(args.delays)+1)
    for row in table_data:
        print(print_holder.format(*row))
