import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def distribution(data, numeric_cols, categorical_cols):
    # Calculate the number of rows and columns for the subplot grid
    num_plots = len(numeric_cols)
    num_cols = 3  
    num_rows = (num_plots + num_cols - 1) // num_cols  
    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7))
    # Flatten the array of axes and remove any excess axes if the number of plots is not a perfect multiple
    axs = axs.flatten()[:num_plots]
    # Plot histograms for numeric columns
    for ax, col in zip(axs, numeric_cols):
        data[col].hist(ax=ax)
        ax.set_title(col)
    for ax in axs[num_plots:]:
        ax.remove()

    fig.suptitle('Distribution of Numeric Features', fontsize=16, fontweight = "bold")
    # Add a rectangle around the entire figure
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.00, 0.0), 1, 1, fill=False, color="k", lw=2, 
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    # Add the rectangle to the figure
    fig.patches.extend([rect])
    # Apply tight layout to adjust for the histograms
    plt.tight_layout()
    # Show the plot
    plt.show()
    fig.savefig(r'.\data\num_distribution.png', bbox_inches="tight")
    plt.close(fig)

def heat_map(data, numeric_cols):
    ## create heatmap to check features correlation
    fig = plt.figure(figsize=(14,7))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap= "Blues", annot_kws={"size": 14})
    fig.suptitle('Correlation Heatmap of Numeric Features', fontsize=16, fontweight = "bold")
    # Add a rectangle around the entire figure
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.00, 0.0), 1, 1, fill=False, color="k", lw=2, 
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    # Add the rectangle to the figure
    fig.patches.extend([rect])
    plt.tight_layout()
    plt.show()
    fig.savefig(r'.\data\num_feature_correlation.png', bbox_inches="tight")
    plt.close(fig)

def read_file_explor(df):
    data = df.copy()
    ## save describe data to 2 csv (one for categorical one for num data)
    data.describe().T.to_csv(r'.\data\num_stat_data.csv')
    data.describe(include=['O']).T.to_csv(r'.\data\cat_stat_data.csv')
    ## define num, cat colunms
    categorical_cols = data.select_dtypes('object').columns
    numeric_cols = [col for col in data.columns if col not in categorical_cols]
    ## num features distribution
    distribution(data, numeric_cols=numeric_cols, categorical_cols= categorical_cols)
    heat_map(data, numeric_cols)

if __name__ == '__main__':
    df = pd.read_csv('.\data\credit.csv')
    read_file_explor(df)