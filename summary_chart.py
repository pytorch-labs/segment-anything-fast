import pandas as pd
import matplotlib.pyplot as plt

def make_sub_chart(df, ax, title, category_column, value_column, ylim_low, ylim_high):
    ax.bar(df[category_column], df[value_column])
    
    # Customize the chart labels and title
    ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.set_title(title)
    ax.set_ylim(ylim_low, ylim_high)

    tick_positions = ax.get_yticks()
    for tick in tick_positions:
        ax.axhline(y=tick, color='gray', linestyle='--', alpha=0.7)

    

def make_row_chart(df, value_column, ax1, ax2, ylim_low, ylim_high, title, relative=False):
    category_column = "idx"

    vit_b_df = df[df['sam_model_type'] == "vit_b"]
    vit_h_df = df[df['sam_model_type'] == "vit_h"]

    vit_b_df = vit_b_df.copy()
    vit_h_df = vit_h_df.copy()

    if relative:
        vit_b_df[value_column] = vit_b_df[value_column].div(vit_b_df[value_column].iloc[0])
        vit_h_df[value_column] = vit_h_df[value_column].div(vit_h_df[value_column].iloc[0])
    
    make_sub_chart(vit_b_df, ax1, f"{title} for vit_b", category_column, value_column, ylim_low, ylim_high)
    make_sub_chart(vit_h_df, ax2, f"{title} for vit_h", category_column, value_column, ylim_low, ylim_high)

csv_file = "results.csv"
df = pd.read_csv(csv_file)

print(df)
print(df.columns)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
make_row_chart(df, "img_s(avg)", ax1, ax2, 1.0, 2.0, "Speedup", relative=True)
make_row_chart(df, "memory(MiB)", ax3, ax4, 0.0, 1.0, "Memory savings")
plt.tick_params(axis='both', which='both', length=10)
plt.tight_layout()

fig.savefig('bar_chart.svg', format='svg')
