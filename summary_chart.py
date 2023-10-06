import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import fire
import numpy as np

# COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())
COLORS = [(0.7, 0.7, 0.7), (0., 0., 0.), (0.9, 0.9, 0.9)]
# import pdb; pdb.set_trace()


def make_row_chart(df, value_column, ax, label, ylim_low, ylim_high, va, techniques, up_good, up_to, title=None, data_format=None, highlight=False):
    category_column = "technique"

    x_values = list(df[category_column])
    y_values = list(df[value_column])
    # import pdb; pdb.set_trace()
    x_coords = [techniques[x] for x in x_values]
    bar_colors = [COLORS[0] for _ in range(len(x_coords))]
    ax.bar(x_values, y_values, label=label, color=bar_colors)
    if highlight:
        ax.bar(x_values[up_to], y_values[up_to], color="red")

    # Customize the chart labels and title
    # ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.set_title(title)
    ax.set_ylim(ylim_low, ylim_high)

    tick_positions = ax.get_yticks()
    for tick in tick_positions:
        ax.axhline(y=tick, color='gray', linestyle='--', alpha=0.7)

    last_value = None
    difference = None
    # Add data labels or data points above the bars
    for i, (x, value) in enumerate(zip(x_coords, df[value_column])):
        if i > 0:
            difference = value / last_value
        last_value = value
        if va == "top":
            ax.text(x, 0.9 * value, data_format.format(value), ha='center', va=va, color='black')
        else:
            ax.text(x, 0.9 * value, data_format.format(value), ha='center', va=va)
        if x == up_to:
            break
    if difference is not None and difference > 1:
        difference_v = round(difference * 100) - 100
        if up_good:
            ax.text(x, value, f"+{difference_v}%", ha='center', va="bottom", color='green')
        else:
            ax.text(x, value, f"+{difference_v}%", ha='center', va="bottom", color='red')
    if difference is not None and difference == 1:
        ax.text(x, value, f"+/-0%", ha='center', va="bottom", color='gray')
    if difference is not None and difference < 1:
        difference_v = round(difference * 100) - 100
        if up_good:
            ax.text(x, value, f"{difference_v}%", ha='center', va="bottom", color='red')
        else:
            ax.text(x, value, f"{difference_v}%", ha='center', va="bottom", color='green')

            

    tlabels = []
    for i in range(up_to + 1):
        tlabels.append(matplotlib.text.Text(i, 0, list(techniques.keys())[i]))

    # tlabels[2] = matplotlib.text.Text(2, 0, 'compile')
    # tlabels[4] = matplotlib.text.Text(4, 0, 'Triton')
    # if up_to == 8:
    #     tlabels[-1] = matplotlib.text.Text(6, 0, '2:4')
    # tlabels = tlabels[:up_to] + list(map(lambda x: "", tlabels[up_to:]))
    ax.set_xticklabels(tlabels, rotation = 0, ha="center")


def run(up_to):
    matplotlib.rcParams.update({'font.size': 48})
    
    csv_file = "results.csv"
    mdf_ = pd.read_csv(csv_file)
    mdf = mdf_.dropna(subset=["batch_size"])
    techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'int8': 6}
    if up_to == 8:
        techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'sparse': 6}
    techniques = OrderedDict(sorted(techniques.items(), key=lambda kv: kv[1]))
    keys = list(techniques.keys())
    actually_is_8 = False
    if up_to == 8:
        actually_is_8 = True
        up_to = 7
    # mdf = pd.concat([mdf[mdf["technique"] == keys[i]] for i in range(up_to)])
    print("keys: ", keys)

    mdf["memory(GiB)"] = mdf["memory(MiB)"] // 1024
    mdf["img/s"] = mdf["img_s(avg)"].round(decimals=0) // 1
    
    fig, axs = plt.subplots(2, 2, figsize=(40, 18))

    baseline = mdf[mdf["batch_size"] == 1]
    baseline = baseline[baseline["technique"] == "fp32"]
    baseline_vit_b = baseline[baseline["sam_model_type"] == "vit_b"]
    baseline_vit_h = baseline[baseline["sam_model_type"] == "vit_h"]
    other = mdf[mdf["batch_size"] == 32]
    other = other[other["technique"] != "fp32"]
    other_vit_b = other[other["sam_model_type"] == "vit_b"]
    other_vit_h = other[other["sam_model_type"] == "vit_h"]

    other_vit_b = pd.concat([other_vit_b[other_vit_b["technique"] == keys[i]] for i in range(1, 7)])
    other_vit_h = pd.concat([other_vit_h[other_vit_h["technique"] == keys[i]] for i in range(1, 7)])
    
    va = "bottom"
    make_row_chart(baseline_vit_b, "img/s", axs[0][0], "Batch size 1", 0.0, 100.0, va, techniques, True, up_to, "",
                   data_format="{:.0f}")
    make_row_chart(baseline_vit_b, "memory(GiB)", axs[1][0], "Batch size 1", 0, 60, va, techniques, False, up_to, "",
                   data_format="{:.0f}")
    va = "top"
    make_row_chart(other_vit_b, "img/s", axs[0][0], "Batch size 32", 0.0, 100.0, va, techniques, True, up_to, "",
                   data_format="{:.0f}", highlight=True)
    make_row_chart(other_vit_b, "memory(GiB)", axs[1][0], "Batch size 32", 0, 60, va, techniques, False, up_to, "",
                   data_format="{:.0f}", highlight=True)
    # ax2.set_facecolor((252 / 255., 246 / 255., 229 / 255.))

    for ax in axs[1:]:
        ax[0].legend()
        ax[1].legend()
    # plt.tick_params(axis='both', which='both', length=10)
    plt.tight_layout()
    
    if actually_is_8:
        up_to = 8
        fig.savefig(f'bar_chart_{up_to}.svg', format='svg')
    else:
        fig.savefig(f'bar_chart_{up_to}.svg', format='svg')

if __name__ == "__main__":
    fire.Fire(run)
