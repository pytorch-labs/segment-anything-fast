import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())

def make_sub_chart(df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format, label):
    x_values = []
    y_values = []
    bar_colors = []
    x_idx = 0
    for key in techniques.keys():
        if key in df[category_column].tolist():
            x_values.append(key)
            y_values.append(df[value_column].tolist()[x_idx])
            bar_colors.append(COLORS[x_idx])
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
        ax.text(x, value, data_format.format(value), ha='center', va='bottom')


def make_row_chart(df, value_column, ax1, ax2, ax3, label, ylim_low=None, ylim_high=None, title="", relative=False, data_format=None):
    category_column = "technique"

    def helper(sam_model_type, ax1):
        vit_b_df = df[df['sam_model_type'] == sam_model_type]

        vit_b_df = vit_b_df.copy()

        if relative:
            vit_b_df[value_column] = vit_b_df[value_column].div(
                vit_b_df[value_column].iloc[0])

        make_sub_chart(vit_b_df, ax1, f"{title} for {sam_model_type}",
                       category_column, value_column, ylim_low, ylim_high, data_format, label)
    helper("vit_b", ax1)
    helper("vit_l", ax2)
    helper("vit_h", ax3)

matplotlib.rcParams.update({'font.size': 12})

csv_file = "results.csv"
mdf_ = pd.read_csv(csv_file)
mdf = mdf_.dropna(subset=["batch_size"])
techniques = {}
tech_idx = 0
for _, name in enumerate(list(mdf["technique"])):
    if name in techniques:
        pass
    else:
        techniques[name] = tech_idx
        tech_idx += 1
print("techniques: ", techniques)

fig, axs = plt.subplots(3, 3, figsize=(20, 20))

for batch_size in [128, 64, 32, 16, 8, 1]:
    df = mdf[mdf["batch_size"] == batch_size]
    
    # print(df)
    # print(df.columns)
    
    if batch_size in [1, 32]:
        make_row_chart(df, "img_s(avg)", *axs[0], f"Batch size {batch_size}", 0.0, 100.0,
                       "Images per second", data_format="{:.2f}")
    make_row_chart(df, "memory(MiB)", *axs[1], f"Batch size {batch_size}", 0, 80000,
                   title="Memory savings", data_format="{:.0f}")
    if batch_size in [16]:
        make_row_chart(df, "mIoU", *axs[2], f"Batch size {batch_size}", 0.0, 1.0,
                       title="Accuracy", data_format="{:.2f}")
for ax in axs:
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
# plt.tick_params(axis='both', which='both', length=10)
plt.tight_layout()

fig.savefig('bar_chart.svg', format='svg')
