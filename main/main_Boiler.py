import os
import subprocess

if __name__ == '__main__':
    all_files = os.listdir(os.path.join('../datasets', 'Boiler'))

    files = [name[:-4] for name in all_files if name.endswith('.csv')]

    files = sorted(files)
    for src in files:
        for trg in files:
            if src != trg:
                print('src: ', src, ' / target: ', trg)

                train = os.path.join('[path-to-dacad]/main/', 'train.py')
                command = ['python', train, '--algo_name', 'cluda', '--num_epochs', '20', '--queue_size', '98304', '--momentum', '0.99', '--batch_size', '256',
                           '--eval_batch_size', '256', '--learning_rate', '1e-4', '--dropout', '0.2', '--weight_decay', '1e-4', '--num_channels_TCN', '128-128-128',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7', '--hidden_dim_MLP', '256', '--id_src', src, '--id_trg', trg,
                           '--experiment_folder', 'boiler']
                subprocess.run(command)

                test = os.path.join('[path-to-dacad]/main/', 'eval.py')
                command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder', 'boiler', '--id_src', src, '--id_trg', trg]
                subprocess.run(command1)