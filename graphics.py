import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy.io
from plotly.subplots import make_subplots
from numpy import arange

from globals import *

plt.rcParams.update({
    "text.usetex": True,
    "font.sans-serif": ["Helvetica"]})


def load_files():
    data = []
    for mod in signals:
        file_name = pathlib.Path(join(os.getcwd(), 'mat-data', mod + '_features.mat'))
        file = scipy.io.loadmat(file_name)
        data.append(file[mat_info[mod]])
    return data


def calculate_features_mean(data):
    ft_array = np.array(data)
    ft_mean_array = np.ndarray((len(signals), len(testing_SNR), 1, number_of_used_features))
    for i in range(len(signals)):
        for j in range(len(testing_SNR)):
            for k in range(number_of_used_features):
                ft_mean_array[i, j, 0, k] = np.mean(ft_array[i, j, :, k])
    return ft_mean_array


def calculate_features_stddev(data):
    ft_array = np.array(data)
    ft_mean_array = np.ndarray((len(signals), len(testing_SNR), 1, number_of_used_features))
    for i in range(len(signals)):
        for j in range(len(testing_SNR)):
            for k in range(number_of_used_features):
                ft_mean_array[i, j, 0, k] = np.std(ft_array[i, j, :, k])
    return ft_mean_array


def generate_snr_axis():
    snr_values = np.linspace((testing_SNR[0] - 5) * 2, (testing_SNR[-1] - 5) * 2, len(testing_SNR))
    # Repeat x_axis for all modulations in data
    x_axis = np.ndarray((len(signals), len(testing_SNR)))
    for i in range(len(signals)):
        x_axis[i, :] = snr_values
    return x_axis


def simple_plot(snr_axis, data_axis, plot_type='html', save=True):
    if plot_type == 'html':
        # Plot HTML window using PLOTLY
        fig = make_subplots(rows=5, cols=5, subplot_titles=used_features_names)
        R, C = 1, 1
        for ft in range(number_of_used_features):
            if C == 6:
                R += 1
                C = 1
            for label, signal in enumerate(signals):
                if ft == 0:
                    fig.add_trace(go.Scatter(x=snr_axis[label, :],
                                             y=data_axis[label, :, 0, ft],
                                             legendgroup=signal,
                                             name=signal,
                                             line=dict(color=px.colors.qualitative.Plotly[label])), row=R, col=C)
                else:
                    fig.add_trace(go.Scatter(x=snr_axis[label, :],
                                             y=data_axis[label, :, 0, ft],
                                             legendgroup=signal,
                                             name=signal,
                                             showlegend=False,
                                             line=dict(color=px.colors.qualitative.Plotly[label])), row=R, col=C)

            C += 1
        fig.update_layout(width=1920 * 2, height=1080 * 2, legend=dict(
            orientation="h",
            yanchor="auto",
            y=1.05,
            xanchor="auto",
            x=0,
            title_font_family="Arial",
            title="Modulation",
            font=dict(
                family="Arial",
                size=16,
                color="black"
            )
        ))
        fig.show()
        if save:
            figure_name = pathlib.Path(join(os.getcwd(),
                                            'figures',
                                            'features', 'plots.html'))
            fig.write_html(figure_name.__str__())
            del fig
    elif plot_type == 'png':
        # Plot graphics using only mean (matplotlib.plot)
        for n in range(number_of_used_features):
            plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
            plt.plot(snr_axis[0, :], data_axis[0, :, 0, n], '#2F8000', linewidth=1.0, antialiased=True)  # BPSK
            plt.plot(snr_axis[1, :], data_axis[1, :, 0, n], '#DEAA0B', linewidth=1.0, antialiased=True)  # QPSK
            plt.plot(snr_axis[2, :], data_axis[2, :, 0, n], '#FF3300', linewidth=1.0, antialiased=True)  # PSK8
            plt.plot(snr_axis[3, :], data_axis[3, :, 0, n], '#AD00E6', linewidth=1.0, antialiased=True)  # QAM16
            plt.plot(snr_axis[4, :], data_axis[4, :, 0, n], '#0066FF', linewidth=1.0, antialiased=True)  # QAM64
            plt.plot(snr_axis[5, :], data_axis[5, :, 0, n], 'k', linewidth=1.0, antialiased=True)  # Noise
            # plt.title('Feature ' + str(n + 1) + ' - ' + features_names[n])
            plt.xlabel('SNR [dB]')
            plt.xticks(snr_axis[0, :], SNR_values.values())
            plt.ylabel(features_names[n+1], rotation=0, fontsize=15, labelpad=20)
            plt.legend(signals)
            figure_name = pathlib.Path(join(os.getcwd(),
                                            'figures',
                                            'features',
                                            'ft_{}_SNR_({})_a_({})_mean.png'.format(str(n + 1),
                                                                                    (testing_SNR[0] - 5) * 2,
                                                                                    (testing_SNR[-1] - 5) * 2)))
            if save:
                plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                            orientation='landscape', format='png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            print('Plotting means of feature number {}'.format(n))


def n_frames_plot(n_frames, snr_axis, data_axis, save=False):
    # TODO: HTML plot for all frames
    for n in range(number_of_used_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.plot(snr_axis[0, :], data_axis[0, :, 0:n_frames, n], '#2F8000', linewidth=1.0, antialiased=True)  # BPSK
        plt.plot(snr_axis[1, :], data_axis[1, :, 0:n_frames, n], '#DEAA0B', linewidth=1.0, antialiased=True)  # QPSK
        plt.plot(snr_axis[2, :], data_axis[2, :, 0:n_frames, n], '#FF3300', linewidth=1.0, antialiased=True)  # PSK8
        plt.plot(snr_axis[3, :], data_axis[3, :, 0:n_frames, n], '#AD00E6', linewidth=1.0, antialiased=True)  # QAM16
        plt.plot(snr_axis[4, :], data_axis[4, :, 0:n_frames, n], '#0066FF', linewidth=1.0, antialiased=True)  # QAM64
        plt.plot(snr_axis[5, :], data_axis[5, :, 0:n_frames, n], 'k', linewidth=1.0, antialiased=True)  # Noise
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_axis[0, :], SNR_values.values())
        plt.ylabel(features_names[n+1], rotation=0, fontsize=15, labelpad=20)
        BPSK_patch = mpatches.Patch(color='#2F8000', label='BPSK')
        QPSK_patch = mpatches.Patch(color='#DEAA0B', label='QPSK')
        PSK8_patch = mpatches.Patch(color='#FF3300', label='PSK8')
        QAM16_patch = mpatches.Patch(color='#AD00E6', label='QAM16')
        QAM64_patch = mpatches.Patch(color='#0066FF', label='QAM64')
        noise_patch = mpatches.Patch(color='k', label='noise')
        plt.legend(handles=[BPSK_patch, QPSK_patch, PSK8_patch, QAM16_patch, QAM64_patch, noise_patch])
        # plt.title('Feature ' + str(n + 1) + ' - ' + features_names[n])

        figure_name = pathlib.Path(join(os.getcwd(),
                                        'figures',
                                        'features',
                                        'ft_{}_SNR_({})_a_({})_{}_frames.png'.format(str(n + 1),
                                                                                     (testing_SNR[0] - 5) * 2,
                                                                                     (testing_SNR[-1] - 5) * 2,
                                                                                     n_frames)))
        if save:
            plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                        orientation='landscape', format='png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        print('Plotting n frames of feature number {}'.format(n))


def errorbar_plot(snr_axis, mean, stddev, save=False):
    # TODO: HTML plot for errorbar
    # Plot graphics with error bar using standard deviation
    for n in range(number_of_used_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.errorbar(snr_axis[0, :],
                     mean[0, :, 0, n],
                     yerr=stddev[0, :, 0, n], color='#2F8000', linewidth=1.0)
        plt.errorbar(snr_axis[1, :],
                     mean[1, :, 0, n],
                     yerr=stddev[1, :, 0, n], color='#DEAA0B', linewidth=1.0)
        plt.errorbar(snr_axis[2, :],
                     mean[2, :, 0, n],
                     yerr=stddev[2, :, 0, n], color='#FF3300', linewidth=1.0)
        plt.errorbar(snr_axis[3, :],
                     mean[3, :, 0, n],
                     yerr=stddev[3, :, 0, n], color='#AD00E6', linewidth=1.0)
        plt.errorbar(snr_axis[4, :],
                     mean[4, :, 0, n],
                     yerr=stddev[4, :, 0, n], color='#0066FF', linewidth=1.0)
        plt.errorbar(snr_axis[5, :],
                     mean[5, :, 0, n],
                     yerr=stddev[5, :, 0, n], color='k', linewidth=1.0)
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_axis[0, :], SNR_values.values())
        plt.ylabel(features_names[n+1], rotation=0, fontsize=15, labelpad=20)
        # plt.title('Feature ' + str(n + 1) + ' - ' + features_names[n])
        plt.legend(signals)
        figure_name = pathlib.Path(join(os.getcwd(),
                                        'figures',
                                        'features',
                                        'ft_{}_SNR_({})_a_({})_err.png'.format(str(n + 1),
                                                                               (testing_SNR[0] - 5) * 2,
                                                                               (testing_SNR[-1] - 5) * 2)))
        if save:
            plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                        orientation='landscape', format='png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        print('Plotting error bar of feature number {}'.format(n))


if __name__ == '__main__':
    # Load files
    files = load_files()
    # Process
    snr_array = generate_snr_axis()
    mean_array = calculate_features_mean(files)
    std_array = calculate_features_stddev(files)
    # Plot
    simple_plot(snr_array, mean_array, plot_type='png', save=True)
    n_frames_plot(100, snr_array, np.array(files), save=True)
    # errorbar_plot(snr_array, mean_array, std_array, save=True)
