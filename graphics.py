import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy.io
from plotly.subplots import make_subplots

from globals import *

# Code to use LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.sans-serif": ["Helvetica"]})


# Function to load files
def load_files():
    data = []
    for mod in modulation_signals:
        file_name = calculated_features_folder.joinpath(mod + '_features.mat').__str__()
        file = scipy.io.loadmat(file_name)
        print(file_name + ' file loaded...')
        data.append(file[mat_info[mod]])
    return data


# This function is used to generate the errorbar plot
def calculate_mean_and_stddev(data):
    ft_array = np.array(data)
    ft_mean_array = np.ndarray((len(modulation_signals), len(plotting_snr), 1, len(used_features)))
    ft_stddev_array = np.ndarray((len(modulation_signals), len(plotting_snr), 1, len(used_features)))
    for i in range(len(modulation_signals)):
        for j in range(len(plotting_snr)):
            for k in range(len(used_features)):
                ft_mean_array[i, j, 0, k] = np.mean(ft_array[i, j, :, k])
                ft_stddev_array[i, j, 0, k] = np.std(ft_array[i, j, :, k])
    return ft_mean_array, ft_stddev_array


def generate_snr_axis():
    snr_axis = np.linspace((plotting_snr[0] - 5) * 2, (plotting_snr[-1] - 5) * 2, len(plotting_snr))
    # Repeat x_axis for all modulations in data
    x_axis = np.ndarray((len(modulation_signals), len(plotting_snr)))
    for i in range(len(modulation_signals)):
        x_axis[i, :] = snr_axis
    return x_axis


# This function can be used to generate the HTML plot with all features or the separated PNG files for each feature
# You can also choose to save the file or just plot it on a window
def simple_plot(snr_axis, data_axis, plot_type='png', save=True):
    if plot_type == 'html':
        # Plot HTML window using PLOTLY
        fig = make_subplots(rows=5, cols=5, subplot_titles=used_features_names)
        rows, cols = 1, 1
        for ft in range(len(used_features)):
            if cols == 6:
                rows += 1
                cols = 1
            for label, signal in enumerate(modulation_signals):
                if ft == 0:
                    fig.add_trace(go.Scatter(x=snr_axis[label, :],
                                             y=data_axis[label, :, 0, ft],
                                             legendgroup=signal,
                                             name=signal,
                                             line=dict(color=px.colors.qualitative.Plotly[label])), row=rows, col=cols)
                else:
                    fig.add_trace(go.Scatter(x=snr_axis[label, :],
                                             y=data_axis[label, :, 0, ft],
                                             legendgroup=signal,
                                             name=signal,
                                             showlegend=False,
                                             line=dict(color=px.colors.qualitative.Plotly[label])), row=rows, col=cols)

            cols += 1
        fig.update_layout(width=1920 * 2, height=1080 * 2,
                          legend=dict(orientation="h", yanchor="auto", y=1.05, xanchor="auto", x=0,
                                      title_font_family="Arial", title="Modulation",
                                      font=dict(family="Arial", size=16, color="black")))
        fig.show()
        if save:
            figure_name = feature_figures_folder.joinpath('all_plots.html')
            fig.write_html(figure_name.__str__())
            del fig
    elif plot_type == 'png':
        # Plot graphics using only mean (matplotlib.plot)
        for n in range(len(used_features)):
            plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
            plt.plot(snr_axis[0, :], data_axis[0, :, 0, n], '#2F8000', linewidth=1.0, antialiased=True)  # BPSK
            plt.plot(snr_axis[1, :], data_axis[1, :, 0, n], '#DEAA0B', linewidth=1.0, antialiased=True)  # QPSK
            plt.plot(snr_axis[2, :], data_axis[2, :, 0, n], '#FF3300', linewidth=1.0, antialiased=True)  # 8PSK
            plt.plot(snr_axis[3, :], data_axis[3, :, 0, n], '#AD00E6', linewidth=1.0, antialiased=True)  # 16QAM
            plt.plot(snr_axis[4, :], data_axis[4, :, 0, n], '#0066FF', linewidth=1.0, antialiased=True)  # 64QAM
            plt.xlabel('SNR [dB]')
            plt.xticks(snr_axis[0, :], snr_values.values())
            plt.ylabel(features_names[n + 1], rotation=0, fontsize=15, labelpad=20)
            plt.legend(modulation_signals)
            filename_str = 'ft{}_SNR({}to{})_mean.png'.format(str(n + 1),
                                                              int((plotting_snr[0] - 5) * 2),
                                                              int((plotting_snr[-1] - 5) * 2))
            figure_name = feature_figures_folder.joinpath(filename_str)
            if save:
                plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                            orientation='landscape', format='png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            print('Plotting means of feature number {}'.format(used_features[n]))


# You can choose to save the file or just plot it on a window
# This functions can run slow if too many frames are plotted
def multiple_frames_plot(n_frames, snr_axis, data_axis, save=False):
    # TODO: HTML plot for all frames
    for n in range(len(used_features)):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.plot(snr_axis[0, :], data_axis[0, :, 0:n_frames, n], '#2F8000', linewidth=1.0, antialiased=True)  # BPSK
        plt.plot(snr_axis[1, :], data_axis[1, :, 0:n_frames, n], '#DEAA0B', linewidth=1.0, antialiased=True)  # QPSK
        plt.plot(snr_axis[2, :], data_axis[2, :, 0:n_frames, n], '#FF3300', linewidth=1.0, antialiased=True)  # 8PSK
        plt.plot(snr_axis[3, :], data_axis[3, :, 0:n_frames, n], '#AD00E6', linewidth=1.0, antialiased=True)  # 16QAM
        plt.plot(snr_axis[4, :], data_axis[4, :, 0:n_frames, n], '#0066FF', linewidth=1.0, antialiased=True)  # 64QAM
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_axis[0, :], snr_values.values())
        plt.ylabel(features_names[n + 1], rotation=0, fontsize=15, labelpad=20)
        bpsk_patch = mpatches.Patch(color='#2F8000', label='BPSK')
        qpsk_patch = mpatches.Patch(color='#DEAA0B', label='QPSK')
        _8_psk_patch = mpatches.Patch(color='#FF3300', label='8PSK')
        _16_qam_patch = mpatches.Patch(color='#AD00E6', label='16QAM')
        _64_qam_patch = mpatches.Patch(color='#0066FF', label='64QAM')
        plt.legend(handles=[bpsk_patch, qpsk_patch, _8_psk_patch, _16_qam_patch, _64_qam_patch])
        filename_str = 'ft{}_SNR({}to{})_{}frames.png'.format(str(n + 1),
                                                              int((plotting_snr[0] - 5) * 2),
                                                              int((plotting_snr[-1] - 5) * 2),
                                                              n_frames)
        figure_name = feature_figures_folder.joinpath(filename_str)
        if save:
            plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                        orientation='landscape', format='png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        print('Plotting n frames of feature number {}'.format(used_features[n]))


# You can choose to save the file or just plot it on a window
def errorbar_plot(snr_axis, mean, stddev, save=False):
    # TODO: HTML plot for errorbar
    # Plot graphics with error bar using standard deviation
    for n in range(len(used_features)):
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
        plt.xlabel('SNR [dB]')
        plt.xticks(snr_axis[0, :], snr_values.values())
        plt.ylabel(features_names[n + 1], rotation=0, fontsize=15, labelpad=20)
        plt.legend(modulation_signals)
        filename_str = 'ft{}_SNR({}to{})_err.png'.format(str(n + 1),
                                                         int((plotting_snr[0] - 5) * 2),
                                                         int((plotting_snr[-1] - 5) * 2))
        figure_name = feature_figures_folder.joinpath(filename_str)
        if save:
            plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                        orientation='landscape', format='png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        print('Plotting error bar of feature number {}'.format(used_features[n]))


def plot():
    # Load files
    data = load_files()

    # Generate axis and calculate means and standard deviations
    snr_array = generate_snr_axis()
    mean_array, stddev_array = calculate_mean_and_stddev(data)

    # Plot the default config and save files
    simple_plot(snr_array, mean_array, save=True)
    multiple_frames_plot(100, snr_array, np.array(data), save=True)
    errorbar_plot(snr_array, mean_array, stddev_array, save=True)

    print('Done plotting!')
