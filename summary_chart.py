import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def make_sub_chart(df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format):
    ax.bar(df[category_column], df[value_column])

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
    for i, value in enumerate(df[value_column]):
        ax.text(i, value, data_format.format(value), ha='center', va='bottom')


def make_row_chart(df, value_column, ax1, ax2, ax3, ylim_low=None, ylim_high=None, title="", relative=False, data_format=None):
    category_column = "technique"

    def helper(sam_model_type, ax1):
        vit_b_df = df[df['sam_model_type'] == sam_model_type]

        vit_b_df = vit_b_df.copy()

        if relative:
            vit_b_df[value_column] = vit_b_df[value_column].div(
                vit_b_df[value_column].iloc[0])

        make_sub_chart(vit_b_df, ax1, f"{title} for {sam_model_type}",
                       category_column, value_column, ylim_low, ylim_high, data_format)
    helper("vit_b", ax1)
    helper("vit_l", ax2)
    helper("vit_h", ax3)

matplotlib.rcParams.update({'font.size': 12})

csv_file = "results.csv"
df = pd.read_csv(csv_file)

print(df)
print(df.columns)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)
      ) = plt.subplots(3, 3, figsize=(20, 20))
make_row_chart(df, "img_s(avg)", ax1, ax2, ax3, 0.0, 100.0,
               "Images per second", data_format="{:.2f}")
make_row_chart(df, "memory(MiB)", ax4, ax5, ax6,
               title="Memory savings", data_format="{:.0f}")
make_row_chart(df, "mIoU", ax7, ax8, ax9, 0.0, 1.0,
               title="Accuracy", data_format="{:.2f}")
# plt.tick_params(axis='both', which='both', length=10)
plt.tight_layout()

fig.savefig('bar_chart.svg', format='svg')
