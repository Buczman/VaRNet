from caviar import caviar_prediction
from garch import garch_prediction
from garch_benchmark import garch_benchmark
from caviar_benchmark import caviar_benchmark
import torch

training_sample = 1000
testing_sample = 250
epochs_per_step = 200
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_starts = [
    '2005-01-01',
    '2007-01-01',
    '2013-01-01',
    '2016-01-01'
]

indexes = [
    'wig',
    'spx',
    'lse'
]

memory_sizes = [
    30,
    10,
    5
]

# 1 (początek poniżej, koniec: 2009-12-31)
# [1] "2006-01-23 - ger"
# [1] "2006-01-12 - usa"
# [1] "2005-12-05 - jap"
# [1] "2005-11-24 - chi"
# [1] "2006-01-05 - pol"
# [1] "2005-12-08 - rus"
# 2 (początek poniżej, koniec: 2011-12-31)
# [1] "2008-01-31 - ger"
# [1] "2008-01-15 - usa"
# [1] "2007-11-28 - jap"
# [1] "2007-11-27 - chi"
# [1] "2008-01-11 - pol"
# [1] "2007-12-07 - rus"
# 3 (początek poniżej, koniec: 2017-12-31)
# [1] "2014-01-20 - ger"
# [1] "2014-01-13 - usa"
# [1] "2013-12-02 - jap"
# [1] "2013-11-29 - chi"
# [1] "2014-01-03 - pol"
# [1] "2014-01-10 - rus"
# 4 (początek poniżej, koniec: 2020-10-01)
# [1] "2016-10-17 - ger"
# [1] "2016-10-12 - usa"
# [1] "2016-08-25 - jap"
# [1] "2016-08-24 - chi"
# [1] "2016-09-29 - pol"
# [1] "2016-10-14 - rus"

for sample_start in sample_starts:
    for index in indexes:
        for memory_size in memory_sizes:
            # caviar_prediction(index, sample_start, training_sample, testing_sample, memory_size, epochs_per_step,
            #                   batch_size, device, True)
            # caviar_prediction(index, sample_start, training_sample, testing_sample, memory_size, epochs_per_step,
            #                   batch_size, device, False)
            garch_prediction(index, sample_start, training_sample, testing_sample, memory_size, epochs_per_step,
                             batch_size, device, 'normal')
            garch_prediction(index, sample_start, training_sample, testing_sample, memory_size, epochs_per_step,
                             batch_size, device, 'skewstudent')
            garch_benchmark(index, sample_start, training_sample, testing_sample, memory_size, 'skewstudent')
            garch_benchmark(index, sample_start, training_sample, testing_sample, memory_size, 'normal')
        caviar_benchmark(index, sample_start, training_sample, testing_sample, 1)
