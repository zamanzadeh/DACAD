import os
import subprocess
import numpy as np
import pandas as pd

if __name__ == '__main__':
    all_files = os.listdir(os.path.join('../datasets', 'MSL_SMAP/test'))

    all_names = [name[:-4] for name in all_files if name.endswith('.npy')]
    with open(os.path.join('../datasets/MSL_SMAP/', 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = pd.read_csv(file, delimiter=',')
    data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
    space_files = np.asarray(data_info['chan_id'])

    files = [file for file in all_names if file in space_files]
    files = sorted(files)
    for src in ['F-5', 'C-1', 'D-14', 'P-10']:
        for trg in files:
            if src != trg:
                print('src: ', src, ' / target: ', trg)

                train = os.path.join('[path-to-dacad]/main/', 'train.py')  #
                command = ['python', train, '--algo_name', 'dacad', '--num_epochs', '20', '--queue_size', '98304',
                           '--momentum', '0.99', '--batch_size', '256', '--eval_batch_size', '256', '--learning_rate',
                           '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7', '--hidden_dim_MLP', '1024',
                           '--id_src', src, '--id_trg', trg, '--experiment_folder', 'MSL']
                subprocess.run(command)

                test = os.path.join('[path-to-dacad]/main/', 'eval.py')
                command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder',
                            'MSL', '--id_src', src, '--id_trg', trg]
                subprocess.run(command1)
