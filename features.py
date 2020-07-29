import json
import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib
import os
from os.path import join

from tqdm import tqdm

def calculate_features(input_signal):
    with open("./info.json") as handle:
        info_json = json.load(handle)

    result = []

    for ft in info_json['features']['using']:
        aux = eval(info_json['features']['functions'][str(ft)])
        result.append(aux)

    return result

def plot_ft_histogram(option):
    with open("./info.json") as handle:
        info_json = json.load(handle)

    modulations = info_json['modulations']['names'] 
    number_of_frames = 1000#info_json['numberOfFrames']
    number_of_features = 22#len(info_json['features']['using'])
    features_files = [f + "_features.pickle" for f in modulations]
    features_names = info_json['features']['names']
    features_using = info_json['features']['using']
    snr_list = info_json['snr']['using']
    snr_values = info_json['snr']['values']       
    
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))

    if option == 1:
        for i, mod in enumerate(features_files):
            print("Processing {} data".format(mod.split("_")[0]))
            with open(join(data_folder, mod), 'rb') as ft_handle:
                data = pickle.load(ft_handle)

            report = open("report_" + str(mod.split("_")[0]) + ".txt", 'w+')

            for j,snr in enumerate(snr_list):
                for feature in range(number_of_features):
                    ft = []
                    for frame in range(number_of_frames):
                        ft.append(data[j][frame][feature])
                    
                    """ n, bins, patches = plt.hist(x=ft, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                    plt.grid(axis='y', alpha=0.75)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of {} \nfor SNR {}dB of {}'.format(features_names[feature], snr_values[snr], mod))
                    maxfreq = n.max()
                    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                    plt.savefig(join(fig_folder, 'histogram', 'histogram_specific_{}_{}dB_{}'.format(feature,snr_values[snr], mod.split('_')[0]) + '.png'), bbox_inches='tight', dpi=300)
                    plt.clf() """
                    
                    report.write("Feature:{} - SNR:{}dB - Max:{} - Min:{}\n".format(feature,snr_values[snr], str(max(ft)), str(min(ft))))

            report.close()

    elif option == 2:
        for i, mod in enumerate(features_files):
            print("Processing {} data".format(mod.split("_")[0]))
            with open(join(data_folder, mod), 'rb') as ft_handle:
                data = pickle.load(ft_handle)

            for j, snr in enumerate(snr_list):
                ft = []
                for feature in range(number_of_features):
                    aux = []
                    for frame in range(number_of_frames):
                        aux.append(data[j][frame][feature])
                    ft.append(aux)
                
                for k, element in enumerate(ft):
                    n, bins, patches = plt.hist(x=ft[k], bins='auto', alpha=0.7, rwidth=0.85)
                    plt.grid(axis='y', alpha=0.75)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of all features \nfor SNR {}dB of {}'.format(snr_values[snr], mod))
                    plt.legend(features_using)
                    maxfreq = n.max()
                    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)                    
                    
                plt.savefig(join(fig_folder, 'histogram', 'histogram_all_{}_{}dB_{}'.format(feature,snr_values[snr], mod.split('_')[0]) + '.png'), bbox_inches='tight', dpi=300)
                plt.clf()
    elif option == 3:
        for feature in range(number_of_features):
            print("Processing feature {}".format(feature))            
            for j, snr in enumerate(snr_list):
                ft = []
                for i, mod in enumerate(features_files):
                    aux = []
                    with open(join(data_folder, mod), 'rb') as ft_handle:
                        data = pickle.load(ft_handle)
                    for frame in range(number_of_frames):
                        aux.append(data[j][frame][feature])
                    ft.append(aux)

                for k, element in enumerate(ft):
                    n, bins, patches = plt.hist(x=ft[k],bins='auto', alpha=0.7, rwidth=0.85)
                    plt.grid(axis='y', alpha=0.75)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of feature {} \nfor SNR {}dB'.format(features_names[feature],snr_values[snr]))                    
                    maxfreq = n.max()
                    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                plt.legend(modulations)
                plt.savefig(join(fig_folder, 'histogram', 'histogram_comparison_{}_{}dB'.format(feature,snr_values[snr]) + '.png'), bbox_inches='tight', dpi=300)
                #plt.plot()
                #plt.show()
                plt.clf()
    else:
        print("Invalid option")

if __name__ == '__main__':
    #Option 1 will create a graphic to each feature X snr X modulation
    #Option 2 will create a grapchic with all features X snr X modulation
    #OPtion 3 will create a graphic to each feature x snr, grouped by modulation
    plot_ft_histogram(1)