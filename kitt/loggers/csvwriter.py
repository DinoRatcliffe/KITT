import csv
import os

import numpy as np

class CSVWriter():
    def __init__(self, directory, buffer_size=0):
        self._buffer_size = buffer_size
        self._buffer = {}
        self._directory = directory


    def _flush(self):
        """
        Flushes the buffer out to the appropriate files
        """
        for path, data in self._buffer.items():
            with open(path, 'a') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        self._buffer = {}


    def scalar(self, key, value, epoch):
        key = key.replace('/', '-')
        dir_path = os.path.join(self._directory, 'scalar')
        file_path = os.path.join(dir_path, f'{key}.csv')

        # create directory and file if needed
        if file_path not in self._buffer:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if not os.path.isfile(file_path):
                with open(file_path, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Epoch', 'Value'])

        # write value to buffer and flush to file if needed
        if file_path in self._buffer:
            self._buffer[file_path].append([epoch, value])
        else:
            self._buffer[file_path] = [[epoch, value]]

        if len(self._buffer[file_path]) > self._buffer_size:
            self._flush()

    # TODO (ratcliffe@dino.ai): Save properly just output mean and std at the moment
    def histogram(self, key, values, epoch):
        values_mean = np.mean(values)
        values_stddev = np.std(values)

        key = key.replace('/', '-')
        dir_path = os.path.join(self._directory, 'histogram')
        file_path = os.path.join(dir_path, f'{key}.csv')

        # create directory and file if needed
        if file_path not in self._buffer:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if not os.path.isfile(file_path):
                with open(file_path, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Epoch', 'Mean', 'STDDEV'])

        # write value to buffer and flush to file if needed
        if file_path in self._buffer:
            self._buffer[file_path].append([epoch, values_mean, values_stddev])
        else:
            self._buffer[file_path] = [[epoch, values_mean, values_stddev]]

        if len(self._buffer[file_path]) > self._buffer_size:
            self._flush()
