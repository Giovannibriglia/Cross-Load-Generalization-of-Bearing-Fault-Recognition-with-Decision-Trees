import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fftpack import fft, fftfreq
warnings.filterwarnings('ignore')

sample_rate = 25600
seconds_of_acquistion = 10
path_inputs = 'Vibr_MONO_NO_offset'
path_saving = 'dataframes_mono'
fontsize = 12
labelsize = 12


def notch_filter(vet, notch_freq, quality):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality, sample_rate)

    outputSignal = signal.filtfilt(b_notch, a_notch, vet)

    return outputSignal


def FFT(vet, time_in_sec):
    n = sample_rate * time_in_sec
    T = 1 / sample_rate
    y = vet
    yf = fft(y)
    yfl = 2 * abs(yf[:n // 2] / n)
    xf = fftfreq(n, T)
    xfl = xf[:n // 2]

    return yfl, xfl


def normalize_column(col):
    min_val = col.min()
    max_val = col.max()
    normalized_column = (col - min_val) / (max_val - min_val)

    return normalized_column


features_names = ['mean', 'max', 'kurt', 'skew', 'mad', 'percScore', 'entropy', 'score_at_perc', 'coef_var',
                  'coef_kstatvar', 'min', 'std_class', 'var']
os.makedirs(path_saving, exist_ok=True)

for max_frequency in [375, 125, 75]:
    for n_classes in [2, 3]:

        features = ['name_signal']
        for value in [25, 50]:
            for feature in features_names:
                features.append(feature + f'_10_{value}')

        for start_freq in range(25, max_frequency - 25, 25):
            for feature in features_names:
                features.append(feature + f'_{start_freq}_{start_freq + 50}')

        features.insert(len(features) + 1, 'D_class')

        df = pd.DataFrame([[0] * len(features)], columns=features)

        row_number = 0
        half_sample_rate = int(sample_rate / 2)
        values_for_fft_plots = []


        for filename in glob.glob(f"{path_inputs}\*.csv"):
            with open(os.path.join(os.getcwd(), filename), "r") as f:
                data = pd.read_csv(filename)
                name = filename.replace(f'{path_inputs}\\', '')
                name = name.replace('.csv', '')
                print(name, '-', n_classes, 'classes - until ', max_frequency, ' Hz')
                indexD = int(name[name.index('D') + 1])
                indexR = int(name[name.index('R') + 1])
                indexT = int(name[name.index('T') + 1])

                # *******************************************************************

                for count in range(half_sample_rate, len(data) + half_sample_rate, half_sample_rate):

                    if count == half_sample_rate and indexT + indexR + indexD == 3:
                        if_vis = True
                    else:
                        if_vis = False

                    in1 = count - int(half_sample_rate)
                    in2 = count + int(half_sample_rate)

                    input_data = data.iloc[in1:in2, 0]
                    mean_input_data = np.mean(input_data)
                    input_data = [s-mean_input_data for s in input_data]

                    vet_Notch = input_data.copy()
                    for notch_frequency in range(50, 550, 50):
                        vet_Notch = notch_filter(vet=vet_Notch, notch_freq=notch_frequency, quality=10.0)

                    if if_vis:
                        fig = plt.figure(dpi=600)
                        fig.suptitle(f'Notch Filters on D{indexD}-R{indexR}-T{indexT}', fontsize=fontsize + 5)
                        plt.plot(input_data[:100], label='Before notch filters', linewidth=2)
                        plt.plot(vet_Notch[:100], label='After notch filters', linewidth=2)
                        plt.legend(loc='best')
                        plt.xlabel('Time', fontsize=fontsize)
                        plt.ylabel('Amplitude', fontsize=fontsize)
                        plt.tick_params(axis='x', labelsize=labelsize)
                        plt.tick_params(axis='y', labelsize=labelsize)
                        plt.grid()


                    yfl, xfl = FFT(vet=vet_Notch, time_in_sec=1)
                    if count == half_sample_rate and (indexR + indexT == 2):
                        values_for_fft_plots.append(yfl)
                        yfl_wrong, xfl_wrong = FFT(vet=input_data, time_in_sec=1)
                        values_for_fft_plots.append(yfl_wrong)

                    if if_vis:
                        yfl_wrong, xfl_wrong = FFT(vet=input_data, time_in_sec=1)

                        fig = plt.figure(dpi=600)
                        fig.suptitle(f'FFT on D{indexD}-R{indexR}-T{indexT}', fontsize=fontsize + 5)
                        plt.plot(xfl_wrong[:500], yfl_wrong[:500], label='Before notch filters', linewidth=2)
                        plt.plot(xfl[:500], yfl[:500], label='After notch filters', linewidth=2)
                        plt.legend(loc='best')
                        plt.xlabel('Hz', fontsize=fontsize)
                        plt.ylabel('Amplitude', fontsize=fontsize)
                        plt.tick_params(axis='x', labelsize=labelsize)
                        plt.tick_params(axis='y', labelsize=labelsize)
                        # plt.ylim(0, 1)
                        plt.grid()
                        # plt.savefig('FFT_HealtyCase_NotchFilters.pdf')
                        plt.show()

                    row = []

                    row_name = name + '_' + str(in1) + '_' + str(in2)
                    row.append(row_name)

                    for value in [25, 50]:
                        row.append(np.max(yfl[10:value]))
                        row.append(np.mean(yfl[10:value]))
                        row.append(stats.kurtosis(yfl[10:value]))
                        row.append(stats.skew(yfl[10:value]))
                        row.append(stats.median_abs_deviation(yfl[10:value]))
                        row.append(stats.percentileofscore(yfl[10:value], 0.35))
                        row.append(stats.entropy(yfl[10:value]))
                        row.append(stats.scoreatpercentile(yfl[10:value], 0.35))
                        row.append(stats.variation(yfl[10:value]))
                        row.append(stats.kstatvar(yfl[10:value]))
                        row.append(np.min(yfl[10:value]))
                        row.append(np.std(yfl[10:value]))
                        row.append(np.var(yfl[10:value]))

                    for value in range(25, max_frequency - 25, 25):
                        row.append(np.max(yfl[value:value + 50]))
                        row.append(np.mean(yfl[value:value + 50]))
                        row.append(stats.kurtosis(yfl[value:value + 50]))
                        row.append(stats.skew(yfl[value:value + 50]))
                        row.append(stats.median_abs_deviation(yfl[value:value + 50]))
                        row.append(stats.percentileofscore(yfl[value:value + 50], 0.35))
                        row.append(stats.entropy(yfl[value:value + 50]))
                        row.append(stats.scoreatpercentile(yfl[value:value + 50], 0.35))
                        row.append(stats.variation(yfl[value:value + 50]))
                        row.append(stats.kstatvar(yfl[value:value + 50]))
                        row.append(np.min(yfl[value:value + 50]))
                        row.append(np.std(yfl[value:value + 50]))
                        row.append(np.var(yfl[value:value + 50]))

                    if n_classes == 3:
                        row.append(indexD)
                    elif n_classes == 2:
                        if indexD > 1:
                            row.append(1)
                        else:
                            row.append(0)

                    df.loc[row_number] = row
                    row_number += 1

        if max_frequency == 375 and n_classes == 2:
            fig_subplots, axes = plt.subplots(3, 1, sharex=True, dpi=600)
            fig.suptitle('Spectra Resulting Monoaxial Accelerometer', fontsize=fontsize+3)
            x = np.arange(0, 500, 1)
            azz = [': Healthy case', ': Damage on the Outer Ring', ': Brinnelling Damage']
            for val in range(0, len(values_for_fft_plots), 2):
                if int(val / 2) == 0:
                    axes[int(val / 2)].plot(x, values_for_fft_plots[val][:500], linewidth=2, label='After Notch')
                    axes[int(val / 2)].plot(x, values_for_fft_plots[val + 1][:500], linewidth=1, label='Before Notch')
                    axes[int(val / 2)].legend(loc='best')
                else:
                    axes[int(val / 2)].plot(x, values_for_fft_plots[val][:500], linewidth=2)
                    axes[int(val / 2)].plot(x, values_for_fft_plots[val + 1][:500], linewidth=1)

                axes[int(val/2)].set_title(f'D{int(val/2) + 1}-R{indexR}-T{indexT} {azz[int(val/2)]}', fontsize=fontsize + 5)
                axes[int(val/2)].set_ylabel('Amplitude', fontsize=fontsize)
                axes[int(val/2)].grid(True)
            plt.tight_layout()
            plt.savefig('ResultingSignal_Mono.pdf')
            plt.show()

        to_del = []
        for i in range(1, len(features) - 1, 1):
            vet = df.iloc[1:, i]
            std = vet.std()
            if vet.std() == 0:
                to_del.append(i)

        print('Removed features: ', to_del)
        df2 = df.drop(df.columns[to_del], axis=1, inplace=False)

        df2.to_pickle(f'{path_saving}\\{max_frequency}Hz_{n_classes}classes.pkl')

        """
        print('normalizing...')
        features_to_normalize = df2.columns.to_list()[1:-1]
        if 'D_class' in features_to_normalize:
            print('error1')
        if 'name_signal' in features_to_normalize:
            print('error2')
        df2[features_to_normalize] = df2[features_to_normalize].apply(normalize_column)

        df2.to_pickle(f'{path_saving}\\{max_frequency}Hz_{n_classes}classes_normalized.pkl')
        """
