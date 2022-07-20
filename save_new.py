#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 数据采集
from __future__ import print_function
import numpy as np
from numpy import *
import time
from pymoduleconnector import create_mc
import threading
import scipy.io as sio
tmp1 = []
tmp1 = np.array(tmp1)


def radar(com,name):
    # count=8

    # print(count)
    with create_mc(com) as mc:
        xep = mc.get_xep()

        # inti x4driver
        xep.x4driver_init()

        # Set enable pin
        xep.x4driver_set_enable(1)

        # Set iterations
        xep.x4driver_set_iterations(64)
        # Set pulses per step
        xep.x4driver_set_pulses_per_step(5)
        # Set dac step
        xep.x4driver_set_dac_step(1)
        # Set dac min
        xep.x4driver_set_dac_min(949)
        # Set dac max
        xep.x4driver_set_dac_max(1100)
        # Set TX power
        xep.x4driver_set_tx_power(2)

        # Enable downconversion
        xep.x4driver_set_downconversion(0)

        # Set frame area offset
        xep.x4driver_set_frame_area_offset(0.18)
        # offset = xep.x4driver_get_frame_area_offset()

        # Set frame area
        xep.x4driver_set_frame_area(0.2, 3)
        # frame_area = xep.x4driver_get_frame_area()

        # Set TX center freq
        xep.x4driver_set_tx_center_frequency(3)

        # Set PRFdiv
        xep.x4driver_set_prf_div(16)
        # prf_div = xep.x4driver_get_prf_div()

        # Start streaming
        xep.x4driver_set_fps(100)

        # fps = xep.x4driver_get_fps()

        def read_frame():
            """Gets frame data from module"""
            d = xep.read_message_data_float()
            frame = np.array(d.data)
            # Convert the resulting frame to a complex array if downconversion is enabled
            return frame

        # Stop streaming

        print( " wait")
        # save1 = np.ones((20*35, 437))
        save1 = np.ones((30 * 100, 437))
        # save1 = np.ones((30 * 100, 437),dtype = complex)
        fc = 7.1e9
        fs = 23.328e9
        phase_filter = np.array([-0.115974720052285,-0.0773345100048285,0.0365253777771683,-0.0412732590195836,0.0284568407504594,-0.0115161214947589,0.0299994926179564,0.00698978275147158,0.0273822446913041,0.00875491552837954,0.0131136766130344,-0.00510650646551428,-0.00946616689891689,-0.0241168278986621,-0.0274525374698975,-0.0322786742036804,-0.0261354557117109,-0.0167384782609816,0.00194348360668676,0.0240446970936053,0.0511262886037737,0.0778124759461410,0.103468064677834,0.123475425164957,0.137013410794562,0.141361791292646,0.137013410794562,0.123475425164957,0.103468064677834,0.0778124759461410,0.0511262886037737,0.0240446970936053,0.00194348360668676,-0.0167384782609816,-0.0261354557117109,-0.0322786742036804,-0.0274525374698975,-0.0241168278986621,-0.00946616689891689,-0.00510650646551428,0.0131136766130344,0.00875491552837954,0.0273822446913041,0.00698978275147158,0.0299994926179564,-0.0115161214947589,0.0284568407504594,-0.0412732590195836,0.0365253777771683,-0.0773345100048285,-0.115974720052285])

        for jj in range(30 * 100):
            print(jj)
            frame2 = read_frame()
            csine = np.exp(-1j * fc / fs * 2 * np.pi * np.arange(frame2.shape[0]))
            cframe = frame2 * csine
            cframe_lp = np.convolve(phase_filter, cframe)[25:-25]
            phasesignal = np.arctan(np.imag(cframe_lp) / np.real(cframe_lp))
            save1[jj, 0:436] = phasesignal
            ll2 = time.asctime(time.localtime(time.time()))

            ll21 = ll2[11:13]
            ll22 = ll2[14:16]
            ll23 = ll2[17:19]

            lll = int(ll21) * 10000 + int(ll22) * 100 + int(ll23)
            save1[jj, -1] = lll

        sio.savemat('./0519/20220519_'+name+'_mly_uwb_01.mat',{'phasesignal':save1})
        # count+=1
        print('radar!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        xep.module_reset()




if __name__=='__main__':
    com1='COM3'
    com2 = 'COM9'
    name1='radar1'
    name2 = 'radar2'
    t1=threading.Thread(name='radar',target=radar,args=(com1,name1))
    # t2 = threading.Thread(name='radar', target=radar, args=(com2,name2))
    # t2=threading.Thread(name='video',target=video)

    t1.start()
    #t2.start()



