from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    ## FOR WADI and SWAT DATASET
    #df = df.set_index('datetime')

    if 'attack' in df.columns:
        df = df.drop(columns=['attack'])
    
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values.astype("datetime64[ns]") - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    if '.pkl' in args.train_path:
        train = pd.DataFrame(pd.read_pickle(args.train_path))
        test = pd.DataFrame(pd.read_pickle(args.test_path))
        df = train.append([test]).reset_index(drop=True)
        num_test = len(test)
    else:
        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)
        df = train.append([test]).reset_index(drop=True)
        num_test = len(test)
    #df = df.round(6)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-(args.window_size-1), 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 2, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    train_samples = num_samples - num_test
    num_train = round(train_samples * (1-args.val_ratio))
    num_val = train_samples - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = (
        x[-num_test: ],
        y[-num_test: ],
    )
    for cat in ["train", "val","test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)

    labels = pd.read_pickle(args.anomaly_path).tolist()

    with open(args.output_dir+'/anomaly_labels.txt','w') as f:
        f.write(','.join([str(int(i)) for i in labels]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="../data/machine-1-1", help="Output directory."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="machine-1-1_train.pkl",
        help="Train data.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="machine-1-1_test.pkl",
        help="Test data.",
    )
    parser.add_argument(
        "--anomaly_path",
        type=str,
        default="machine-1-1_test_label.pkl",
        help="Anomaly labels"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Input window size for TS forecasting"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.3,
        help="Validation ratio"
    )
    args = parser.parse_args()
    main(args)
