import pandas as pd
import argparse
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers.csv_logs import CSVLogger
from scipy.stats import norm
import skewstudent
from varnet.dataloader import ValueAtRiskDataModule, TimeseriesDataset
from varnet.loss import *
from varnet.nets import VaRNet, SkewedGARCHVaRNet, GARCHVaRNet

import matplotlib.pyplot as plt


def experiment(model_name):

    if model_name == 'garch_skew':
        model_class = SkewedGARCHVaRNet
        dist = skewstudent.skewstudent.SkewStudent
        loss = hansen_garch_skewed_student_loss
    elif model_name == 'garch_norm':
        model_class = GARCHVaRNet
        dist = norm
        loss = garch_normal_loss
    elif model_name == 'caviar':
        model_class = VaRNet
        dist = None
        loss = caviar_loss
    elif model_name == 'caviar_huber':
        model_class = VaRNet
        dist = None
        loss = huber_loss
    else:
        return


    sample_starts = [
        '2005-01-01',
        '2007-01-01',
        '2013-01-01',
        '2016-01-01'
    ]

    indexes = [
        'wig',
        # 'spx',
        # 'lse'
    ]

    memory_sizes = [
        5,
        # 10,
        # 20,
        # 100
    ]
    seed_everything(1)

    for mem_size in memory_sizes:
        p = dict(
            training_length=1000,
            seq_len=mem_size,
            batch_size=512,
            criterion=loss,
            max_epochs=300,
            n_features=1,
            hidden_size=100,
            num_layers=1,
            dropout=0,
            learning_rate=3e-4,
            num_train=250
        )


        for sample_start in sample_starts:
            path = './data/wig.csv'
            data = pd.read_csv(
                path,
                sep=',',
                index_col='Data'
            )
            data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
            data = data.loc[(data.index > sample_start)]
            data['VaR'] = np.nan

            dm = ValueAtRiskDataModule(
                df=data,
                training_length=p['training_length'],
                seq_len=p['seq_len'],
                batch_size=p['batch_size'],
            )

            for test_case in range(p['num_train']):
                csv_logger = CSVLogger('./runs/', name='{}_{}_{}'.format(model_name, mem_size, sample_start), version=str(test_case))
                trainer = Trainer(
                    max_epochs=p['max_epochs'],
                    logger=csv_logger,
                    gpus=1,
                    progress_bar_refresh_rate=20,
                    weights_summary=None
                )

                model = model_class(
                    n_features=p['n_features'],
                    hidden_size=p['hidden_size'],
                    seq_len=p['seq_len'],
                    batch_size=p['batch_size'],
                    criterion=p['criterion'],
                    num_layers=p['num_layers'],
                    dropout=p['dropout'],
                    learning_rate=p['learning_rate'],
                    dist=dist
                )

                dm.setup_train()
                trainer.fit(model, datamodule=dm)
                # model.eval()
                # train_test = TimeseriesDataset(dm.X_train, dm.y_train, 10)
                # pred_out = []
                # true_out = []
                # for i in range(len(train_test)):
                #     pred_out.append(model.forward(torch.tensor(train_test[i][0], dtype=torch.float).unsqueeze(0)).data)
                #     true_out.append(train_test[i][1].data)
                #
                # plt.plot(pred_out)
                # plt.plot(true_out)
                # plt.show()

                # train_test = TimeseriesDataset(dm.X_train, dm.y_train, 100)
                # pred_out = []
                # true_out = []
                # for i in range(len(train_test)):
                #     VaR = model.forward(torch.tensor(train_test[i][0], dtype=torch.float).unsqueeze(0)).detach().numpy()[0]
                #     # dist = skewstudent.skewstudent.SkewStudent(eta=VaR[2], lam=VaR[1])
                #     # var = np.sqrt(VaR[0]) * dist.ppf(0.025)
                #     var = np.sqrt(VaR[0]) * norm().ppf(0.025)
                #     pred_out.append(var)
                #     true_out.append(train_test[i][1].data)
                #
                # plt.plot(pred_out)
                # plt.plot(true_out)
                # plt.show()

                dm.gather_prediction(dm.preprocessing.inverse_transform(model.predict_var(dm.X_test))[0][0])
                dm.move_timestep()

            dm.df[['log_returns', 'VaR']].to_csv('{}_{}_{}.csv'.format(model_name, sample_start, mem_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', metavar='m', type=str,
                        help='model name')
    args = parser.parse_args()
    experiment(args.model_name)
