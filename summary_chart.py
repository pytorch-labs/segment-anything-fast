import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import fire

COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())

def make_sub_chart(df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format, label, va, techniques, batch_size_idx):
    x_values = []
    y_values = []
    bar_colors = []
    x_idx = 0
    for key in techniques.keys():
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
    ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.set_title(title)
    if ylim_low is None:
        assert ylim_high is None
    else:
        ax.set_ylim(ylim_low, ylim_high)

    tick_positions = ax.get_yticks()
    for tick in tick_positions:
        ax.axhline(y=tick, color='gray', linestyle='--', alpha=0.7)

    # Add data labels or data points above the bars
    for x, value in zip(x_coords, df[value_column]):
        ax.text(x, value, data_format.format(value), ha='center', va=va)

    ax.set_xticklabels(df[category_column], rotation = 45, ha="right")


def make_row_chart(df, value_column, ax1, ax2, label, ylim_low, ylim_high, va, techniques, batch_size_idx, title="", relative=False, data_format=None):
    category_column = "technique"
    if not isinstance(ylim_low, tuple):
        ylim_low = (ylim_low, ylim_low)
    if not isinstance(ylim_high, tuple):
        ylim_high = (ylim_high, ylim_high)

    def helper(sam_model_type, ax1, ylim_low, ylim_high, va):
        vit_b_df = df[df['sam_model_type'] == sam_model_type]

        vit_b_df = vit_b_df.copy()

        if relative:
            vit_b_df[value_column] = vit_b_df[value_column].div(
                vit_b_df[value_column].iloc[0])

        make_sub_chart(vit_b_df, ax1, f"{title} for {sam_model_type}",
                       category_column, value_column, ylim_low, ylim_high, data_format, label, va, techniques, batch_size_idx)
    helper("vit_b", ax1, ylim_low[0], ylim_high[0], va)
    helper("vit_h", ax2, ylim_low[1], ylim_high[1], va)

def run(up_to):
    matplotlib.rcParams.update({'font.size': 24})
    
    csv_file = "results.csv"
    mdf_ = pd.read_csv(csv_file)
    mdf = mdf_.dropna(subset=["batch_size"])
    techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'int8': 6, 'sparse': 7}
    print("techniques: ", techniques)
    keys = [k for (k, v) in sorted(techniques.items(), key=lambda kv: kv[1])]
    mdf = pd.concat([mdf[mdf["technique"] == keys[i]] for i in range(up_to)])
    print("keys: ", keys)

    mdf["memory(GiB)"] = mdf["memory(MiB)"] // 1024
    mdf["img/s"] = mdf["img_s(avg)"].round()
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    for batch_size_idx, (batch_size, hlim, va) in enumerate(zip([32, 1], [100, 100], ["bottom", "bottom"])):
        df = mdf[mdf["batch_size"] == batch_size]
        make_row_chart(df, "img/s", *axs[0], f"Batch size {batch_size}", (0.0, 0.0), (100.0, 25.0), va, techniques, batch_size_idx,
                       "Images per second", data_format="{:.2f}")
        make_row_chart(df, "memory(GiB)", *axs[1], f"Batch size {batch_size}", 0, 80, va, techniques, batch_size_idx,
                       title="Memory savings", data_format="{:.0f}")
    for ax in axs:
        ax[0].legend()
        ax[1].legend()
    # plt.tick_params(axis='both', which='both', length=10)
    plt.tight_layout()
    
    fig.savefig(f'bar_chart_{up_to}.svg', format='svg')

if __name__ == "__main__":
    fire.Fire(run)
