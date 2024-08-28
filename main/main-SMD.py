import os
import subprocess

if __name__ == '__main__':
    all_files = os.listdir(os.path.join('../datasets', 'SMD/train/'))
    files = [file for file in all_files if file.startswith('machine-')]
    files = sorted(files)
    for src in ['machine-1-1.txt', 'machine-2-3.txt', 'machine-3-7.txt', 'machine-1-5.txt']:
        src = src.replace('machine-', '')
        src = src.replace('.txt', '')
        for trg in files:
            trg = trg.replace('machine-', '')
            trg = trg.replace('.txt', '')
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                train = os.path.join('[path-to-dacad]/main/', 'train.py')
                command = ['python', train, '--algo_name', 'dacad', '--num_epochs', '20', '--queue_size', '98304', '--momentum', '0.99', '--batch_size', '128',
                           '--eval_batch_size', '256', '--learning_rate', '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7', '--hidden_dim_MLP', '1024', '--id_src', src, '--id_trg', trg, '--experiment_folder', 'SMD']
                subprocess.run(command)

                test = os.path.join('[path-to-dacad]/main/', 'eval.py')
                command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder', 'SMD', '--id_src', src, '--id_trg', trg]
                subprocess.run(command1)