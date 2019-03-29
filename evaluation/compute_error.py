import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from util import get_errors

FONT_SIZE_XLABEL= 15
FONT_SIZE_YLABEL= 15
FONT_SIZE_LEGEND = 11.8
FONT_SIZE_TICK = 11.8


def print_usage():
    print('usage: {} REN_9x6x6 max-frame/mean-frame/joint dataset log_file_name_.txt'.format(sys.argv[0]))
    exit(-1)


def draw_error_bar(dataset, errs, eval_names, fig):
    joint_idx = range(22)
    names = ['Wrist', 'TMCP', 'IMCP', 'MMCP', 'RMCP', 'PMCP', 'TPIP', 'TDIP', 'TTIP', 'IPIP', 'IDIP', 'ITIP', 'MPIP', 'MDIP', 'MTIP', 'RPIP', 'RDIP', 'RTIP', 'PPIP', 'PDIP', 'PTIP', 'Mean']
    max_range = 40      # max error value

    eval_num = len(errs)
    bar_range = eval_num + 1
    ax = fig.add_subplot(1,2,1)
    # color map
    values = range(bar_range-1)
    colors_ = ['blue','green','darkred']            # bar colors for the different datasets

    for eval_idx in range(eval_num):
        x = np.arange(eval_idx, bar_range*len(joint_idx), bar_range)
        mean_errs = np.mean(errs[eval_idx], axis=0)
        mean_errs = np.append(mean_errs, np.mean(mean_errs))
        print('mean error: {:.3f}mm --- {}'.format(mean_errs[-1], eval_names[eval_idx]))
        plt.bar(x, mean_errs[joint_idx], label=eval_names[eval_idx], color=colors_[eval_idx])
    x = np.arange(0, bar_range*len(joint_idx), bar_range)
    plt.xticks(x + 0.5*bar_range, names, rotation='vertical')
    plt.ylabel('Mean Error (mm)', fontsize=FONT_SIZE_YLABEL)
    plt.legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    plt.grid(True)
    major_ticks = np.arange(0, max_range+1, 10)
    minor_ticks = np.arange(0, max_range+1, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
    ax.grid(which='major', alpha=0.5, linestyle='--', linewidth=0.3)
    ax.set_ylim(0, max_range)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelsize=FONT_SIZE_TICK)
    plt.subplots_adjust(bottom=0.14)
    fig.tight_layout()


def draw_error_curve(errs, eval_names, metric_type, fig):
    maxthresh = 100
    eval_num = len(errs)
    thresholds = np.arange(0, maxthresh, 1)
    results = np.zeros(thresholds.shape+(eval_num,))

    ax = fig.add_subplot(1,2,2)
    xlabel = 'Mean distance threshold (mm)'
    ylabel = 'Fraction of frames within distance (%)'

    values = range(eval_num)
    if eval_num < 3:
          jet = plt.get_cmap('jet')
    colors_ = ['blue','green','darkred']        # bar colors for the different datasets

    ls = '-'      # line stile; dashed '--'/solid '-'
    for eval_idx in range(eval_num):
        if metric_type == 'mean-frame':
            err = np.mean(errs[eval_idx], axis=1)
        elif  metric_type == 'max-frame':
            err = np.max(errs[eval_idx], axis=1)
            xlabel = 'Maximum allowed distance to GT (mm)'
        elif  metric_type == 'joint':
            err = errs[eval_idx]
            xlabel = 'Distance Threshold (mm)'
            ylabel = 'Fraction of joints within distance (%)'
        err_flat = err.ravel()
        for idx, th in enumerate(thresholds):
            results[idx, eval_idx] = np.where(err_flat <= th)[0].shape[0] * 1.0 / err_flat.shape[0]

        ax.plot(thresholds, results[:, eval_idx]*100, label=eval_names[eval_idx],
                color=colors_[eval_idx], linestyle=ls)
    plt.xlabel(xlabel, fontsize=FONT_SIZE_XLABEL)
    plt.ylabel(ylabel, fontsize=FONT_SIZE_YLABEL)
    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    plt.grid(True)
    major_ticks = np.arange(0, maxthresh+1, 20)
    minor_ticks = np.arange(0, maxthresh+1, 10)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    major_ticks = np.arange(0, 101, 10)
    minor_ticks = np.arange(0, 101, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
    ax.grid(which='major', alpha=0.5, linestyle='--', linewidth=0.3)
    ax.set_xlim(0, maxthresh)
    ax.set_ylim(0, 100)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelsize=FONT_SIZE_TICK)
    fig.tight_layout()



def main():
    if len(sys.argv) < 4:
        print_usage()

    eval_name = sys.argv[1]
    metric_type = sys.argv[2]
    datasets = []
    log_files = []
    eval_errs = []

    for idx in range(3, len(sys.argv), 2):
        dataset = sys.argv[idx]
        log_file = sys.argv[idx+1]
        datasets.append(dataset)
        log_files.append(log_file)
        print(dataset, log_file, eval_name, metric_type)
        if dataset=='rgbd':
            err = get_errors('fpad', log_file, 'test')
        else:
            err = get_errors(dataset, log_file, 'test')
        eval_errs.append(err)

    fig = plt.figure(figsize=(16, 6))
    plt.figure(fig.number)
    draw_error_bar(eval_name, eval_errs, datasets, fig)
    draw_error_curve(eval_errs, datasets, metric_type, fig)
    plt.savefig('../figures/{}_error----.png'.format(metric_type))

    plt.show()


if __name__ == '__main__':
    main()
