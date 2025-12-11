# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Import data
df = pd.read_csv("medical_examination.csv")


# Add BMI column and overweight column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)


# Normalize cholesterol and gluc values
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x <= 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x <= 1 else 1)


# ------------------------------ #
# CATEGORICAL PLOT FUNCTION
# ------------------------------ #
def draw_cat_plot():
    # Melt DataFrame
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
        var_name='variable',
        value_name='value'
    )

    # Group and count
    df_cat_grouped = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Draw the catplot
    chart = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat_grouped,
        kind='bar'
    )

    fig = chart.fig
    fig.savefig("catplot.png")
    return fig


# ------------------------------ #
# HEATMAP FUNCTION
# ------------------------------ #
def draw_heat_map():

    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate correlation
    corr = df_heat.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=.5,
        cmap="coolwarm",
        cbar_kws={"shrink": .5},
        ax=ax
    )

    fig.savefig("heatmap.png")
    return fig


# -------------------------------- #
# RUN FUNCTIONS TO GENERATE AND SAVE PLOTS
# -------------------------------- #

# Uncomment to generate plots automatically:
# draw_cat_plot()
# draw_heat_map()

