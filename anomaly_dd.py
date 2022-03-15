import numpy as np
from sklearn.decomposition import PCA
import util
from tqdm import tqdm


class anomaly_dd():
    def __init__(self, train_obs, val_obs, test_obs,
                 train_forecast, val_forecast, test_forecast,
                 window_length=None, batch_size=512,root_cause=False):
        self.train_obs = train_obs
        self.val_obs = val_obs
        self.test_obs = test_obs
        self.train_forecast = train_forecast
        self.val_forecast = val_forecast
        self.test_forecast = test_forecast
        self.root_cause = root_cause
        if window_length is None:
            self.window_length = len(train_obs)+len(val_obs)
        else:
            self.window_length = window_length
        self.batch_size = batch_size

        if self.root_cause:
            self.val_re_full = None
            self.test_re_full = None

    def pca_model(self, val_error, test_error, dim_size=1):
        pca = PCA(n_components=dim_size, svd_solver='full')
        pca.fit(val_error)

        transf_val_error = pca.inverse_transform(pca.transform(val_error))
        transf_test_error = pca.inverse_transform(pca.transform(test_error))
        val_re_full = np.absolute(transf_val_error - val_error)
        val_re = val_re_full.sum(axis=1)
        test_re_full = np.absolute(transf_test_error - test_error)
        test_re = test_re_full.sum(axis=1)

        return val_re, test_re, val_re_full, test_re_full

    def scorer(self, num_components):
        val_abs = np.absolute(self.val_obs - self.val_forecast)
        full_obs = np.concatenate((self.train_obs, self.val_obs, self.test_obs), axis=0)
        full_forecast = np.concatenate((self.train_forecast, self.val_forecast, self.test_forecast), axis=0)
        full_abs = np.absolute(full_obs, full_forecast)
        # Normalize forecast error deviation
        val_norm = error_normalizer(val_abs)
        test_norm = error_sw_normalizer(full_abs,self.window_length,self.batch_size,len(self.test_obs))

        # PCA reconstruction algorithm to score anomaly of each timepoint
        val_re, test_re, val_re_full, test_re_full = self.pca_model(val_norm, test_norm, num_components)

        if self.root_cause:
            self.val_re_full = val_re_full
            self.test_re_full = test_re_full

        # Real Time Indicator and Automatic Classifier
        realtime_indicator = test_re
        anomaly_prediction = test_re > val_re.max()

        return realtime_indicator, anomaly_prediction




## function to normalize errors
def error_normalizer(error_mat):
    # Calculate normalization of error
    median = np.median(error_mat, axis=0)
    q1 = np.quantile(error_mat, q=0.25, axis=0)
    q3 = np.quantile(error_mat, q=0.75, axis=0)
    iqr = q3 - q1 + 1e-2  # 1e-2 for numerical stability
    norm_error = (error_mat - median) / iqr

    return norm_error


def sliding_window(error_mat, window_size):
    return error_mat[np.arange(window_size)[None, :] + np.arange(error_mat.shape[0] - window_size)[:, None]]


def error_sw_normalizer(error_mat,window_size,batch_size,test_size):
    data_size = error_mat.shape[0]
    num_batch = int(test_size / batch_size) + 1
    norm_error_mat = []

    # Batch processing
    print('Batch processing to normalize errors')
    for i in tqdm(range(num_batch)):
        # start and end index process for test data
        start_idx = i * batch_size + (data_size - test_size)
        end_idx = (i + 1) * batch_size + (data_size - test_size)
        # batch error and sliding window error
        batch_error_mat = error_mat[start_idx:end_idx, :]
        sw_error_mat = sliding_window(error_mat[(start_idx - window_size):end_idx, :], window_size)

        # Calculate normalization of error
        median = np.median(sw_error_mat, axis=1)
        q1 = np.quantile(sw_error_mat, q=0.25, axis=1)
        q3 = np.quantile(sw_error_mat, q=0.75, axis=1)
        iqr = q3 - q1 + 1e-2    # 1e-2 for numerical stability
        batch_norm_error = (batch_error_mat - median) / iqr
        norm_error_mat.append(batch_norm_error)
    # Concat all batch processing
    norm_error_mat = np.concatenate(norm_error_mat)

    return norm_error_mat

