""" Loads touchscreen performance corpora into dataframes and numpy arrays suitable for training an MDRNN. """
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import h5py
import random


class TinyPerformanceLoader(object):
    """Manages a touchscreen performance corpus and can export sequence or dataframe versions or save to an h5 file."""

    def __init__(self, perf_location="../performances/", verbose=True):
        """
        Loads tiny performance data into lists of dataframes.
        """
        self.verbose = verbose
        performance_dict = {}
        self.logs = []
        self.log_arrays = []
        if self.verbose:
            print("Loading tiny performances...")
        for local_file in os.listdir(perf_location):
            if local_file.endswith(".csv"):
                perf_data = {
                    "title": local_file[:-4],
                    "performer": local_file.split("-")[1],
                    "instrument": local_file.split("-")[2],
                    "filename": perf_location + local_file}
                # parse performance
                perf_log = pd.DataFrame.from_csv(perf_data["filename"], parse_dates=False)
                # calculate velocity
                perf_log["velocity"] = self.create_velocity_column(perf_log)
                # arrange time
                perf_log['time'] = perf_log.index
                perf_log['delta_t'] = perf_log.time.diff()
                perf_log.delta_t = perf_log.delta_t.fillna(0)
                # add to archive
                self.logs.append(perf_log)
                self.log_arrays.append(np.array(perf_log[['x', 'y', 'delta_t', 'moving']]))

                # Centroid
                perf_data["centroid_X"] = perf_log["x"].mean()
                perf_data["centroid_Y"] = perf_log["y"].mean()

                # Centroid S.D.
                perf_data["centroid_X_SD"] = perf_log["x"].std()
                perf_data["centroid_Y_SD"] = perf_log["y"].std()

                # Starting/Ending coordinate
                perf_data["first_X"] = perf_log["x"].iloc[0]
                perf_data["first_Y"] = perf_log["y"].iloc[0]
                perf_data["last_X"] = perf_log["x"].iloc[-1]
                perf_data["last_Y"] = perf_log["y"].iloc[-1]

                # Length
                perf_data["duration"] = perf_log.index[-1] - perf_log.index[0]

                # Percentage of moving touches
                perf_data["percent_moving"] = perf_log["moving"].mean()

                # Mean Velocity
                perf_data["mean_velocity"] = perf_log["velocity"].mean()
                performance_dict.update({perf_data["title"]: perf_data})

                # Number of records
                perf_data["total"] = perf_log["x"].count()

        self.performances = pd.DataFrame.from_dict(performance_dict, orient='index')
        # self.performances.instrument = performances.instrument.astype('category')
        if self.verbose:
            print("Finished loading performances:")
            print(self.performances.describe())

    def single_sequence_corpus(self):
        """
        Returns logs as a single sequence (cleaned).
        """
        return self.clean_log_arrays(self.log_arrays)

    def create_velocity_column(self, perf):
        """
        Adds a velocity column to performance data by
        calculating the distance from previous point
        divided by time.
        """
        perf["velocity"] = perf.index
        perf["velocity"] = perf.moving * np.sqrt(np.power(perf.x.diff(), 2) + np.power(perf.y.diff(), 2)) / perf.velocity.diff()
        perf["velocity"] = perf["velocity"].fillna(0)
        return perf["velocity"]

    def clean_log_arrays(self, logs):
        """
        Concatenates and cleans log arrays, returning a single sequence in a,b,dt format.
        """
        concat_log = (np.concatenate(logs)).T
        log_dict = {'a': concat_log[0], 'b': concat_log[1], 'dt': concat_log[2], 'm': concat_log[3]}
        log_df = pd.DataFrame.from_dict(log_dict)  # make into DF again for cleaning
        # Tiny Performances time defined to be in [0,5.0], thus set limits
        log_df.set_value(log_df[log_df.dt > 5].index, 'dt', 5.0)
        log_df.set_value(log_df[log_df.dt < 0].index, 'dt', 0.0)
        # Tiny Performance bounds defined to be in [[0,1],[0,1]], edit to fix this.
        log_df.set_value(log_df[log_df.a > 1].index, 'a', 1.0)
        log_df.set_value(log_df[log_df.a < 0].index, 'a', 0.0)
        log_df.set_value(log_df[log_df.b > 1].index, 'b', 1.0)
        log_df.set_value(log_df[log_df.b < 0].index, 'b', 0.0)
        if self.verbose:
            # Check values:
            print("\ndescriptions of log values:")
            print("\nall logs:")
            print(log_df.describe())
            print("\ndescription of taps:")
            # As a rule of thumb, could classify taps with dt>0.1 as taps, dt<0.1 as moving touches.
            print(log_df[log_df.m == 0].describe())
            print("\ndescription of moving touches:")
            print(log_df[log_df.m == 1].describe())
        return np.array(log_df[['a', 'b', 'dt']])

    def choose_performance(self):
        perf = random.choice(self.log_arrays)
        perf_t = perf.T
        log_dict = {'a': perf_t[0], 'b': perf_t[1], 'dt': perf_t[2], 'm': perf_t[3]}
        log_df = pd.DataFrame.from_dict(log_dict)  # make into DF again for cleaning
        # Tiny Performances time defined to be in [0,5.0], thus set limits
        log_df.set_value(log_df[log_df.dt > 5].index, 'dt', 5.0)
        log_df.set_value(log_df[log_df.dt < 0].index, 'dt', 0.0)
        # Tiny Performance bounds defined to be in [[0,1],[0,1]], edit to fix this.
        log_df.set_value(log_df[log_df.a > 1].index, 'a', 1.0)
        log_df.set_value(log_df[log_df.a < 0].index, 'a', 0.0)
        log_df.set_value(log_df[log_df.b > 1].index, 'b', 1.0)
        log_df.set_value(log_df[log_df.b < 0].index, 'b', 0.0)
        return np.array(log_df[['a', 'b', 'dt']])

    def sample_without_replacement(self, n=100):
        """Returns n performances sampled randomly from the dataset without replacement."""
        np.random.shuffle(self.logs)
        output = []
        for i in range(n):
            perf = self.logs[i]
            output.append(perf)
        return output

    def archive_corpus(self):
        """Archive performances as an h5 file."""
        total_perf_array = self.single_sequence_corpus()
        if self.verbose:
            print(total_perf_array.shape)
        data_file_name = "TinyPerformanceCorpus.h5"
        with h5py.File(data_file_name, 'w') as data_file:
            data_file.create_dataset('total_performances', data=total_perf_array, dtype='float32')


def main():
    """Load up the default touchscreen performance corpus and export as an h5 file."""
    data_loader = TinyPerformanceLoader()
    data_loader.archive_corpus()

if __name__ == '__main__':
    main()
