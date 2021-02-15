from caviar import caviar_prediction
from garch import garch_prediction
from garch_benchmark import garch_benchmark
from caviar_benchmark import caviar_benchmark
import torch


training_sample = 1000
in_model_testing_sample = 0
testing_sample = 250
epochs_per_step = 300
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sample_starts = [
    # '2005-01-01',
    # '2007-01-01',
    # '2013-01-01',
    '2016-01-01'
]

indexes = [
    'wig',
    # 'spx',
    # 'lse'
]

memory_sizes = [
    # 20,
    10,
    # 5
]

for sample_start in sample_starts:
    for index in indexes:
        for memory_size in memory_sizes:
            caviar_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample, memory_size, epochs_per_step,
                              batch_size, device, True)
            # caviar_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample, memory_size, epochs_per_step,
            #                   batch_size, device, False)
            # garch_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample, memory_size, epochs_per_step,
            #                  batch_size, device, 'normal')
            # garch_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample, memory_size, epochs_per_step,
            #                  batch_size, device, 'skewstudent')
            garch_benchmark(index, sample_start, training_sample, testing_sample, memory_size, 'skewstudent')
            garch_benchmark(index, sample_start, training_sample, testing_sample, memory_size, 'normal')
        # caviar_benchmark(index, sample_start, training_sample, testing_sample, 1)
