#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Dataset Generator
# Author: Ronny Milléo
# GNU Radio version: 3.7.13.5
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy


class gr_data(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Dataset Generator")

        ##################################################
        # Variables
        ##################################################
        self.samples_per_symbol = samples_per_symbol = 8
        self.samp_rate = samp_rate = 100000
        self.mags = mags = [1.0, 0.8, 0.3]
        self.fD = fD = 1
        self.delays = delays = [0.0, 0.9, 1.7]
        self.SNR_linear = SNR_linear = [100,63.0957,39.8107,25.1189,15.8489,10,6.3096,3.9811,2.5119,1.5849,1,0.6310,0.3981,0.2512,0.1585,0.1,0.0631,0.0398,0.0251,0.0158,0.01]
        self.SNR_index = SNR_index = 2

        self.QPSK = QPSK = digital.constellation_qpsk().base()


        self.QAM16 = QAM16 = digital.constellation_16qam().base()


        self.PSK8 = PSK8 = digital.constellation_8psk().base()


        self.BPSK = BPSK = digital.constellation_bpsk().base()


        ##################################################
        # Blocks
        ##################################################
        self.digital_constellation_modulator_0 = digital.generic_mod(
          constellation=PSK8,
          differential=True,
          samples_per_symbol=samples_per_symbol,
          pre_diff_code=True,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.channels_dynamic_channel_model_0 = channels.dynamic_channel_model( samp_rate, 0.01, 1e3, 0.01, 1e3, 8, fD, False, 4.0, (0.0,0.1,1.3), (1,0.99,0.97), 8, SNR_linear[SNR_index], 0 )
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\gr-data\\binary_PSK8(-16)', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_divide_xx_0 = blocks.divide_cc(1)
        self.analog_random_source_x_1 = blocks.vector_source_b(map(int, numpy.random.randint(0, 2, 1024*100*3)), False)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, SNR_linear[SNR_index] + 1)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.analog_random_source_x_1, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.channels_dynamic_channel_model_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.channels_dynamic_channel_model_0, 0))

    def get_samples_per_symbol(self):
        return self.samples_per_symbol

    def set_samples_per_symbol(self, samples_per_symbol):
        self.samples_per_symbol = samples_per_symbol

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.channels_dynamic_channel_model_0.set_samp_rate(self.samp_rate)

    def get_mags(self):
        return self.mags

    def set_mags(self, mags):
        self.mags = mags

    def get_fD(self):
        return self.fD

    def set_fD(self, fD):
        self.fD = fD
        self.channels_dynamic_channel_model_0.set_doppler_freq(self.fD)

    def get_delays(self):
        return self.delays

    def set_delays(self, delays):
        self.delays = delays

    def get_SNR_linear(self):
        return self.SNR_linear

    def set_SNR_linear(self, SNR_linear):
        self.SNR_linear = SNR_linear
        self.channels_dynamic_channel_model_0.set_noise_amp(self.SNR_linear[self.SNR_index])
        self.analog_const_source_x_0.set_offset(self.SNR_linear[self.SNR_index] + 1)

    def get_SNR_index(self):
        return self.SNR_index

    def set_SNR_index(self, SNR_index):
        self.SNR_index = SNR_index
        self.channels_dynamic_channel_model_0.set_noise_amp(self.SNR_linear[self.SNR_index])
        self.analog_const_source_x_0.set_offset(self.SNR_linear[self.SNR_index] + 1)

    def get_QPSK(self):
        return self.QPSK

    def set_QPSK(self, QPSK):
        self.QPSK = QPSK

    def get_QAM16(self):
        return self.QAM16

    def set_QAM16(self, QAM16):
        self.QAM16 = QAM16

    def get_PSK8(self):
        return self.PSK8

    def set_PSK8(self, PSK8):
        self.PSK8 = PSK8

    def get_BPSK(self):
        return self.BPSK

    def set_BPSK(self, BPSK):
        self.BPSK = BPSK


def main(top_block_cls=gr_data, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == '__main__':
    main()
