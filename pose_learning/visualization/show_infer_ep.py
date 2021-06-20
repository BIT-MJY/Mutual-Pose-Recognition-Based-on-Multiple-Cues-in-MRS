# Developed by Junyi Ma
# This file is covered by the LICENSE file in the root of this project.
# Brief: A tool to show the results of error propagation.

import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import matplotlib.dates as mdate
import numpy as np
import random
 


def show_pred_ep(ep_preds_all, var_all,y_true, channels,show_all):
 
    plt.cla()
    plt.close() 

    var_all = np.sqrt(var_all)

    if not show_all:

        data_len = ep_preds_all.shape[0]

        for channel in channels:

            xlims = [0,data_len]
            x_ = range(data_len)
            fig, ax = plt.subplots(figsize=(21,9))
            ydata = ep_preds_all[:,channel]
            ub1 = ep_preds_all[:,channel] + 1*var_all[:,channel]
            lb1 = ep_preds_all[:,channel] - 1*var_all[:,channel]
            ub2 = ep_preds_all[:,channel] + 2*var_all[:,channel]
            lb2 = ep_preds_all[:,channel] - 2*var_all[:,channel]

            ax.plot(x_, ep_preds_all[:,channel], color='orange', label = 'Prediction', linewidth=1.5)
            # ax.plot(x_, ep_preds_all[:,channel] + 1*var_all[:,channel], color='black', label = 'Upper Bound1', linewidth=0.8)
            # ax.plot(x_, ep_preds_all[:,channel] - 1*var_all[:,channel], color='black', label = 'Lower Bound1', linewidth=0.8)
            # ax.plot(x_, ep_preds_all[:,channel] + 2*var_all[:,channel], color='black', label = 'Upper Bound2', linewidth=0.5)
            # ax.plot(x_, ep_preds_all[:,channel] - 2*var_all[:,channel], color='black', label = 'Lower Bound2', linewidth=0.5)
            ax.plot(x_, ep_preds_all[:,channel] + 1*var_all[:,channel], color='xkcd:blue', label = '$\sigma$', linewidth=0.8)
            ax.plot(x_, ep_preds_all[:,channel] - 1*var_all[:,channel], color='xkcd:blue', linewidth=0.8)
            ax.plot(x_, ep_preds_all[:,channel] + 2*var_all[:,channel], color='xkcd:light blue', label = '$2\sigma$', linewidth=0.5)
            ax.plot(x_, ep_preds_all[:,channel] - 2*var_all[:,channel], color='xkcd:light blue', linewidth=0.5)
            ax.plot(x_, y_true[:,channel], color='red', label = 'Ground Truth', linewidth=1)
            ax.fill_between(x_,ydata, ub1, color='xkcd:blue', interpolate=True)
            ax.fill_between(x_,lb1, ydata, color='xkcd:blue', interpolate=True)
            ax.fill_between(x_,ub1, ub2, color='xkcd:light blue', interpolate=True)
            ax.fill_between(x_,lb2, lb1, color='xkcd:light blue', interpolate=True)

            # plt.yticks(color = 'gray')
            # plt.xticks(color = 'gray')
            plt.rc('font',family='Times New Roman') 
            fontdict = {"family":"Times New Roman", 'size':30, 'color':'black'} #Times New Roman, Arial
            plt.xlabel("Index", fontdict = fontdict)
            if channel==0:
                add_str = "$t_x$"
            elif channel == 1:
                add_str = "$t_y$"
            elif channel == 2:
                add_str = "$t_z$"
            elif channel == 3:
                add_str = "$q_1$"
            elif channel == 4:
                add_str = "$q_2$"
            elif channel == 5:
                add_str = "$q_3$"
            elif channel == 6:
                add_str = "$q_0$"
            plt.ylabel("Estimated "+ add_str, fontdict = fontdict)
            plt.yticks(fontproperties = 'Times New Roman', size = 28)
            plt.xticks(fontproperties = 'Times New Roman', size = 28)
            ax.spines['top'].set_visible(False) 
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('lightgray')
            plt.grid(color = 'lightgray', linestyle = '-', linewidth = 0.5)
            # plt.grid(axis = 'y', color = 'lightgray', linestyle = '-', linewidth = 0.5)
            leg = plt.legend(loc = 'best', fontsize = 30)
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()
            plt.setp(leg_lines, linewidth=3)
            # plt.setp(leg_texts, fontsize='x-large')

            plt.savefig("img/pose_" + str(channel) + ".png" )
            plt.savefig("img/pose_" + str(channel) + ".eps" )

            # plt.show()

    else: # show_all

        data_len = ep_preds_all.shape[0]

        plt.figure(figsize=(21,9))
        fig, ax = plt.subplots(7, figsize=(21,9))
        for channel in range(7):
            xlims = [0,data_len]
            x_ = range(data_len)
            ydata = ep_preds_all[:,channel]
            ub1 = ep_preds_all[:,channel] + 1*var_all[:,channel]
            lb1 = ep_preds_all[:,channel] - 1*var_all[:,channel]
            ub2 = ep_preds_all[:,channel] + 2*var_all[:,channel]
            lb2 = ep_preds_all[:,channel] - 2*var_all[:,channel]

            ax[channel].plot(x_, ep_preds_all[:,channel], color='orange', label = 'Prediction', linewidth=1.5)
            ax[channel].plot(x_, ep_preds_all[:,channel] + 1*var_all[:,channel], color='black', label = 'Upper Bound1', linewidth=0.8)
            ax[channel].plot(x_, ep_preds_all[:,channel] - 1*var_all[:,channel], color='black', label = 'Lower Bound1', linewidth=0.8)
            ax[channel].plot(x_, ep_preds_all[:,channel] + 2*var_all[:,channel], color='black', label = 'Upper Bound2', linewidth=0.5)
            ax[channel].plot(x_, ep_preds_all[:,channel] - 2*var_all[:,channel], color='black', label = 'Lower Bound2', linewidth=0.5)
            ax[channel].plot(x_, y_true[:,channel], color='red', label = 'GT', linewidth=0.5)
            ax[channel].fill_between(x_,ydata, ub1, color='xkcd:blue', interpolate=True)
            ax[channel].fill_between(x_,lb1, ydata, color='xkcd:blue', interpolate=True)
            ax[channel].fill_between(x_,ub1, ub2, color='xkcd:light blue', interpolate=True)
            ax[channel].fill_between(x_,lb2, lb1, color='xkcd:light blue', interpolate=True)

            # plt.yticks(color = 'gray')
            # plt.xticks(color = 'gray')
            fontdict = {"family":"Times New Roman", 'size':12, 'color':'black'} #Times New Roman, Arial
            plt.xlabel("Index", fontdict = fontdict)
            plt.ylabel("Prediction", fontdict = fontdict)
            # ax[channel].spines['top'].set_visible(False) 
            # ax[channel].spines['left'].set_visible(False)
            # ax[channel].spines['right'].set_visible(False)
            ax[channel].spines['bottom'].set_color('lightgray')
            plt.grid(axis = 'y', color = 'lightgray', linestyle = '-', linewidth = 0.5)
            leg = plt.legend(loc = 'best', fontsize = 12)
            # ax[channel].set_axis_off()
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()
            plt.setp(leg_lines, linewidth=4)
            # plt.setp(leg_texts, fontsize='x-large')
            plt.show()
        plt.show()

