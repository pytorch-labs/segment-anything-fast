import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import fire
import numpy as np

# COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())
COLORS = [(0, 0, 0), (0.3, 0.3, 0.3), (0.6, 0.6, 0.6)]
# import pdb; pdb.set_trace()

def make_sub_chart(df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format, label, va, techniques, batch_size_idx, up_good, up_to):
    x_values = []
    y_values = []
    bar_colors = []
    x_idx = 0
    for i, key in enumerate(techniques.keys()):
        if key in df[category_column].tolist():
            x_values.append(key)
            y_values.append(df[value_column].tolist()[x_idx])
            bar_colors.append(COLORS[batch_size_idx])
            x_idx += 1
        else:
            x_values.append(key)
            y_values.append(0)
    x_coords = [techniques[name] for name in df[category_column]]
    ax.bar(x_values, y_values, label=label, color=bar_colors)

    # Customize the chart labels and title
    # ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    if title is not None:
        ax.set_title(title)
    if ylim_low is None:
        assert ylim_high is None
    else:
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

            

    tlabels = ax.get_xticklabels()
    tlabels[2] = matplotlib.text.Text(2, 0, 'compile')
    tlabels[4] = matplotlib.text.Text(4, 0, 'Triton')
    if up_to == 8:
        tlabels[-1] = matplotlib.text.Text(6, 0, '2:4')
    tlabels = tlabels[:up_to] + list(map(lambda x: "", tlabels[up_to:]))
    ax.set_xticklabels(tlabels, rotation = 0, ha="center")


def make_row_chart(df, value_column, ax1, ax2, label, ylim_low, ylim_high, va, techniques, batch_size_idx, up_good, up_to, title=None, relative=False, data_format=None):
    category_column = "technique"
    if not isinstance(ylim_low, tuple):
        ylim_low = (ylim_low, ylim_low)
    if not isinstance(ylim_high, tuple):
        ylim_high = (ylim_high, ylim_high)

    def helper(sam_model_type, ax1, ylim_low, ylim_high, va, up_to):
        vit_b_df = df[df['sam_model_type'] == sam_model_type]

        vit_b_df = vit_b_df.copy()

        if relative:
            vit_b_df[value_column] = vit_b_df[value_column].div(
                vit_b_df[value_column].iloc[0])

        make_sub_chart(vit_b_df, ax1, title if title is None else f"{title}{sam_model_type}",
                       category_column, value_column, ylim_low, ylim_high, data_format, label, va, techniques, batch_size_idx, up_good, up_to)
    helper("vit_b", ax1, ylim_low[0], ylim_high[0], va, up_to)
    helper("vit_h", ax2, ylim_low[1], ylim_high[1], va, up_to)
    ax2.set_facecolor((252 / 255., 246 / 255., 229 / 255.))

def run(up_to):
    matplotlib.rcParams.update({'font.size': 48})
    
    csv_file = "results.csv"
    mdf_ = pd.read_csv(csv_file)
    mdf = mdf_.dropna(subset=["batch_size"])
    techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'int8': 6}
    if up_to == 8:
        techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'sparse': 6}
    keys = [k for (k, v) in sorted(techniques.items(), key=lambda kv: kv[1])]
    actually_is_8 = False
    if up_to == 8:
        actually_is_8 = True
        up_to = 7
    mdf = pd.concat([mdf[mdf["technique"] == keys[i]] for i in range(up_to)])
    print("keys: ", keys)

    mdf["memory(GiB)"] = mdf["memory(MiB)"] // 1024
    mdf["img/s"] = mdf["img_s(avg)"].round(decimals=0) // 1
    
    fig, axs = plt.subplots(2, 2, figsize=(40, 18))
    
    for batch_size_idx, (batch_size, hlim, va) in enumerate(zip([32, 1], [100, 100], ["top", "bottom"])):
        df = mdf[mdf["batch_size"] == batch_size]
        make_row_chart(df, "img/s", *axs[0], f"Batch size {batch_size}", (0.0, 0.0), (100.0, 100.0), va, techniques, batch_size_idx, True, up_to, "",
                       data_format="{:.0f}")
        make_row_chart(df, "memory(GiB)", *axs[1], f"Batch size {batch_size}", 0, 60, va, techniques, batch_size_idx, False, up_to, "",
                       data_format="{:.0f}")
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
