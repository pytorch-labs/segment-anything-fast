import pandas as pd
import fire
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

COLORS = [(0.7, 0.7, 0.7), (0., 0., 0.), (0.9, 0.9, 0.9)]

def make_sub_chart(color, techniques, df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format, label, va):
    x_values = []
    y_values = []
    bar_colors = []
    x_idx = 0
    for key in techniques.keys():
        if key in df[category_column].tolist():
            x_values.append(key)
            y_values.append(df[value_column].tolist()[x_idx])
            bar_colors.append(color)
            x_idx += 1
        else:
            x_values.append(key)
            y_values.append(0)
    x_coords = []
    for name in df[category_column]:
        if name in techniques:
            x_coords.append(techniques[name])
    ax.bar(x_values, y_values, label=label, color=color, edgecolor="black")

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


def make_row_chart(sam_model_type, color, techniques, df, value_column, ax, label, ylim_low, ylim_high, va, title, relative=False, data_format=None):
    category_column = "technique"

    vit_b_df = df[df['sam_model_type'] == sam_model_type]

    vit_b_df = vit_b_df.copy()

    if relative:
        vit_b_df[value_column] = vit_b_df[value_column].div(
            vit_b_df[value_column].iloc[0])

    make_sub_chart(color, techniques, vit_b_df, ax, f"{title} for {sam_model_type}",
                   category_column, value_column, ylim_low, ylim_high, data_format, label, va)


METRICS = ('img_s', 'memory', 'mIoU')
MODELS = ('vit_h', 'vit_b')

def run(csv_file,
        fig_format,
        batch_sizes,
        metrics=METRICS,
        models=MODELS,
        fig_name='bar_chart'):

    if not isinstance(batch_sizes, tuple):
        batch_sizes = (batch_sizes,)
    print("batch_sizes: ", batch_sizes)

    if not isinstance(metrics, tuple):
        metrics = (metrics,)
    print("metrics: ", metrics)
    for m in metrics:
        assert m in METRICS

    if not isinstance(models, tuple):
        models = (models,)
    print("models: ", models)
    for m in models:
        assert m in MODELS
    matplotlib.rcParams.update({'font.size': 20})
    
    if csv_file == 'stdin':
        import sys
        import io
        csv_file = io.StringIO(sys.stdin.read())
    mdf_ = pd.read_csv(csv_file)
    mdf = mdf_.dropna(subset=["batch_size"])
    techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'int8': 6, 'sparse': 7}
    print("techniques: ", techniques)
    
    fig, axs = plt.subplots(len(metrics), len(models), figsize=(len(models) * 10, len(metrics) * 8))
    if len(models) == 1:
        axs = np.stack([axs])
    if len(metrics) == 1:
        axs = np.stack([axs])
    
    for batch_size in batch_sizes:
        color = 'black' if (batch_size == 1) else COLORS[2]
        df = mdf[mdf["batch_size"] == batch_size]
        va = "bottom"
        if 'img_s' in metrics:
            if 'vit_b' in models:
                make_row_chart('vit_b', color, techniques, df, "img_s(avg)", axs[0][0], f"Batch size {batch_size}", 0.0, 120.0, va, "Images per second", data_format="{:.2f}")
            if 'vit_h' in models:
                make_row_chart('vit_h', color, techniques, df, "img_s(avg)", axs[0][1], f"Batch size {batch_size}", 0.0, 40.0, va, "Images per second", data_format="{:.2f}")
        if 'memory' in metrics:
            if 'vit_b' in models:
                make_row_chart('vit_b', color, techniques, df, "memory(MiB)", axs[1][0], f"Batch size {batch_size}", 0, 40000, va, "Memory savings", data_format="{:.0f}")
            if 'vit_h' in models:
                make_row_chart('vit_h', color, techniques, df, "memory(MiB)", axs[1][1], f"Batch size {batch_size}", 0, 40000, va, "Memory savings", data_format="{:.0f}")
        if 'mIoU' in metrics:
            if 'vit_b' in models:
                make_row_chart('vit_b', color, techniques, df, "mIoU", axs[2][0], f"Batch size {batch_size}", 0.0, 1.0, va, "Accuracy", data_format="{:.2f}")
            if 'vit_h' in models:
                make_row_chart('vit_h', color, techniques, df, "mIoU", axs[2][1], f"Batch size {batch_size}", 0.0, 1.0, va, "Accuracy", data_format="{:.2f}")
    for ax in axs:
        for a in ax:
            a.legend()
    # plt.tick_params(axis='both', which='both', length=10)
    plt.tight_layout()
    
    fig.savefig(f'{fig_name}.{fig_format}', format=fig_format)

if __name__ == '__main__':
    fire.Fire(run)
