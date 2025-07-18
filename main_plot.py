# Main Plot of the paper - Confidence intervals

from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time_stuff.utils import ActivationDataset
from time_stuff.plot import  plot_activations, plot_activations_single
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

plt.rcParams['text.usetex'] = False

# Define reordered Set1 palette from list of original colors
reordered_colors = sns.color_palette("deep", n_colors=9)
reordered_colors = [reordered_colors[3], reordered_colors[2], reordered_colors[0], ]
reordered_Set1 = sns.color_palette(reordered_colors, n_colors=len(reordered_colors))


def underline(text):
    return ''.join(char + '\u0332' for char in text)

def create_custom_subplot_grid(fig_rows, fig_cols, figsize_multiplier=(5, 4)):
    """
    Create a custom subplot grid where even columns have full width and odd columns have half width.
    """
    
    # Calculate figure size
    width_mult, height_mult = figsize_multiplier
    figsize = (fig_cols * width_mult * 2, fig_rows * height_mult)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.2)  # Increase left margin to fit ylabels

    
    # Create gridspec with 3 grid columns per column pair
    # This allows us to make some subplots span 2/3 (full) and others span 1/3 (half)
    total_grid_cols = fig_cols * 3
    gs = gridspec.GridSpec(fig_rows, total_grid_cols, figure=fig)
    
    # Initialize axes array
    axs = np.empty((fig_rows, fig_cols * 2), dtype=object)
    
    # Create subplots
    for row in range(fig_rows):
        for col in range(fig_cols * 2):
            if col % 2 == 0:  # Even columns (0, 2, 4, ...) - full width
                # Span 2 out of 3 grid columns
                grid_start = (col // 2) * 3
                ax = fig.add_subplot(gs[row, grid_start:grid_start + 2])
            else:  # Odd columns (1, 3, 5, ...) - half width
                # Span 1 out of 3 grid columns
                grid_start = (col // 2) * 3 + 2
                ax = fig.add_subplot(gs[row, grid_start:grid_start + 1])
            
            axs[row, col] = ax
    
    return fig, axs

def add_subplot_separators(fig, axs, direction='vertical', every_n=1, 
                            color='black', linewidth=1.5, alpha=0.3, linestyle='--',
                            margin=(0.05, 0.95)):
    """
    Adds separator lines between subplot groups without affecting layout.
    """
    
    axs = np.atleast_2d(axs)
    nrows, ncols = axs.shape

    if direction == 'vertical':
        for col in range(every_n, ncols, every_n):
            ax = axs[0, col]
            bbox = ax.get_position()
            x = bbox.x0
            line = Line2D([x, x], [margin[0], margin[1]],
                          transform=fig.transFigure,
                          color=color, linewidth=linewidth, alpha=alpha,
                          linestyle=linestyle, zorder=1000, clip_on=False)
            fig.add_artist(line)

    elif direction == 'horizontal':
        for row in range(every_n, nrows, every_n):
            ax = axs[row, 0]
            bbox = ax.get_position()
            y = bbox.y1
            line = Line2D([margin[0], margin[1]], [y, y],
                          transform=fig.transFigure,
                          color=color, linewidth=linewidth, alpha=alpha,
                          linestyle=linestyle, zorder=1000, clip_on=False)
            fig.add_artist(line)
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

def add_shared_title(text, ax1, ax2, fig=None, y_pad=0.02, fontsize=14, **kwargs):
    """
    Adds a shared title above two horizontally adjacent Axes.
    """
    if fig is None:
        fig = ax1.figure

    # Get positions in figure coordinates
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    # Calculate center and top position
    x_center = (pos1.x0 + pos2.x1) / 2
    y_top = max(pos1.y1, pos2.y1)

    # Add text to figure
    fig.text(x_center, y_top + y_pad, text,
             ha='center', va='bottom', fontsize=fontsize, **kwargs)

def clean_scatterplot(ax):
    # 1. Remove ticks and labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


    # 2. Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 3. Optional: lighter background
    # ax.set_facecolor('#f9f9f9')  # subtle off-white

def clean_barplot(ax, subset, order=None):
    # 1. Clean gridless style
    sns.set_style("white")
    # 2. Remove spines (all borders)
    sns.despine(ax=ax, left=True)
   
    # 4. Move labels inside bars
    ax.set_yticklabels([])
    ax.set_ylabel('')
    
    # Use the order list to iterate in the correct sequence
    if order is not None:
        manifolds_to_process = order
    else:
        manifolds_to_process = subset['manifold'].unique()

    top_manifolds = subset.nlargest(3, 'norm_score')['manifold'].tolist()
    
    for i, manifold in enumerate(manifolds_to_process):
        # Get the value for this manifold
        value = subset[subset['manifold'] == manifold]['norm_score'].iloc[0] if len(subset[subset['manifold'] == manifold]) > 0 else 0
        midpoint = subset['norm_score'].clip(lower=0).mean()
        maximum = subset['norm_score'].max()

        if value > 0:
            if value == maximum:
                ax.text(value * 0.93, i, underline(manifold), ha='right', va='center', color='white', fontsize=13, fontweight='bold')
            else:
                ax.text(value * 0.93, i, manifold, ha='right', va='center', color='white', fontsize=10, fontweight='bold')

        # Add ranking markers for top 3
        if manifold in top_manifolds:
            rank = top_manifolds.index(manifold) + 1
            marker = f"{rank}°"
            if rank == 1:
                ax.text(value * 1.08, i, marker, ha='left', va='center', 
                    color='dimgray', fontsize=12, fontweight='bold')
            elif rank != 1:
                ax.text(value * 1.08, i, marker, ha='left', va='center', 
                    color='dimgray', fontsize=9, fontweight='bold')
            
        # if value > midpoint:  # Inside the bar
        #     ax.text(value * 0.95, i, manifold, ha='right', va='center', color='white', fontsize=10, fontweight='bold')
        # elif value > 0:  # Outside the bar
        #     ax.text(value * 1.05, i, manifold, ha='left', va='center', color='dimgray', fontsize=10, fontweight='bold')
        # else:
        #     ax.text(0.05, i, manifold, ha='left', va='center', color='dimgray', fontsize=10, fontweight='bold')
    
    # # 5. Reduce x-axis elements
    # ax.set_xlabel('')
    # ax.set_xticks([])
    # ax.set_xticks([0.0, 0.3, 0.7, 1.0])
    # Set font size at 0.8
    ax.tick_params(axis='x', labelsize=12)

df = pd.read_csv('results/scores_scale.csv')
target_col = 'last_prompt_token'  # or 'last_prompt_token' based on your analysis

# Filter dataset
df = df[df['n_components'] == 3]
df = df[df['target_col'] == target_col]
df = df[df['layer'] > 2]
df = df[np.isinf(df['score']) == False]  # Remove inf scores
# df = df[df['target_col'] == 'last_prompt_token']
# df = df[~df['manifold'].isin(['log_semicircular'])]

# Rename manifold trivial to cluster
df['manifold'] = df['manifold'].replace({'trivial': 'cluster', 'euclidean': 'linear'})
df['']

# df = df[df['dataset_name'].isin(['date_3way', 'date_3way_season', 'periodic_3way', 'notable_3way'])] 
# df = df[df['dataset_name'].isin(['date_3way_temperature', 'duration_3way', 'time_of_day_3way', 'time_of_day_3way_phase'])]
df = df[~((df['dataset_name'] == 'date_3way_season') & (df['preprocess_func'] == 'datetime_to_dayofyear'))]
# df = df[df['dataset_name'].isin(['date_3way', 'date_3way_season'])] # Plot 1
# df = df[df['dataset_name'].isin(['periodic_3way', 'notable_3way'])] # Plot 2

### DEBUG ### 
# df = df[df['dataset_name'].isin(['date_3way',])]


# Step 1: For each (model, dataset), get the best (layer, score)
best_layer_manifold = df.loc[df.groupby(['model_name', 'dataset_name'])['score'].idxmax()][['model_name', 'dataset_name', 'layer']]
best_layer_manifold = best_layer_manifold.rename(columns={'layer': 'best_layer'})

# Step 2: Get all the scores for all manifolds but only for the best layer
all_scores = df.merge(best_layer_manifold, on=['model_name', 'dataset_name'])

# Step 2.5: all_scores['fold_scores'] contains a list. Explode it to get individual scores
# First convert the string column into a column of lists
all_scores['fold_score'] = all_scores['fold_scores'].apply(lambda x: eval(x) if isinstance(x, str) else x)
all_scores = all_scores.drop(columns=['fold_scores'])
all_scores = all_scores.explode('fold_score')
all_scores['fold_score'] = all_scores['fold_score'].astype(float)  # Ensure fold_score is float

best_scores = all_scores[all_scores['layer'] == all_scores['best_layer']]



# Step 3: Compute the relative score wrt the best manifold for each (model, dataset)
best_scores['norm_score'] = best_scores.groupby(['model_name', 'dataset_name'])['score'].transform(
    # lambda x: (x - x.min()) / (x.max() - x.min() if x.max() != x.min() else 1)
    # lambda x: x
    # lambda x: (x-x.mean()) / (x.std() if x.std() != 0 else 1)  # Normalize to mean and std
    # lambda x: (x.clip(lower=0) - x.clip(lower=0).min()) / (x.max() - x.clip(lower=0).min() if x.max() != x.clip(lower=0).min() else 
    lambda x: -np.log(1-x)
)
# best_scores['norm_score'] = best_scores.transform(lambda x: -np.log(1-x['score']))
best_scores['norm_fold_score'] = best_scores.groupby(['model_name', 'dataset_name'])['fold_score'].transform(lambda x: -np.log(1-x))


best_scores['manifold_type'] = best_scores['manifold'].map({
    'euclidean': 'linear', 
    'discrete_circular': 'circular', 
    'cluster': 'cluster',
    'circular': 'circular',
    'semicircular': 'linear',
    'log_linear': 'linear',
    'log_semicircular': 'linear'})

# Rename euclidean to linear
best_scores['manifold'] = best_scores['manifold'].replace({'euclidean': 'linear'})

# Step 4: Get the best manifold
best_manifolds = best_scores.loc[best_scores.groupby(['model_name', 'dataset_name'])['norm_score'].idxmax()]
    
best_scores['label_col'] = best_scores['dataset_name'].map({
    'duration_3way': 'correct_duration_length',
    'date_3way': 'correct_date',
    'time_of_day_3way': 'correct_time',
    'notable_3way': 'correct_date',
    'periodic_3way': 'correct_period_length',
})

# best_scores = best_scores[best_scores['model_name'] == 'meta-llama/Llama-3.2-3B-Instruct']
# best_scores = best_scores[best_scores['dataset_name'] == 'date_3way']
best_scores = best_scores[(best_scores['dataset_name'] != 'duration_3way') | (best_scores['preprocess_func'].isna())]
# print(best_scores[['model_name', 'dataset_name', 'manifold', 'score', 'layer']].to_string(index=False))



#### MANIFOLD PLOTS ####
best_manifolds['preprocess_func'] = best_manifolds['dataset_name'].map({
    'duration_3way': None,
    'date_3way': lambda x: pd.to_datetime(x).day_of_year,
    'time_of_day_3way': lambda x: pd.to_datetime(x).hour,
    'notable_3way': lambda x: pd.to_datetime(x).year,
    'periodic_3way': None,
})
best_manifolds['postprocess_func'] = best_manifolds['dataset_name'].map({
    'periodic_3way': lambda x: np.log(x + 1),
    'duration_3way': lambda x: np.log(x + 1),
})
# best_manifolds['category_hue'] = best_manifolds['manifold_type'].map({
#     t:h for t, h in zip(['linear', 'circular', 'cluster'], sns.color_palette(n_colors=3))
#     })
    

def plot_activation_manifold(model_name, dataset_name, layer, manifold, manifold_type, ax, target_col, 
                             label_col=None, preprocess_func=None, postprocess_func=None):
    # Load the activation dataset
    model_name = model_name.split('/')[-1]  # Extract the model name from the full path
    if model_name == 'Llama-3.1-70B-Instruct':
        model_name = '70B-Instruct'
    elif model_name == 'Llama-3.1-8B-Instruct':
        model_name = '8B-Instruct'
    try:
        ad_path = f'results/{model_name}/{dataset_name}.pt'
        ad = ActivationDataset.load(ad_path)
    except FileNotFoundError:
        ad_path = f'results/{model_name}/{dataset_name}'
        ad = ActivationDataset.load(ad_path)
    if manifold_type == 'circular':
        palette = 'twilight'
    elif manifold_type == 'linear':
        palette = 'flare'
    else:
        palette = 'tab10'
    plot_activations_single(
        ad, 
        label_col=label_col,
        target_col=target_col,
        reduction_method='SMDS',
        layer=layer,
        ax=ax,
        manifold=manifold,
        palette=palette,
        preprocess_func=preprocess_func,
        postprocess_func=postprocess_func,
    )

best_manifolds = best_manifolds.sort_values(by=['dataset_name', 'model_name', 'manifold_type', 'norm_score'], ascending=[True, True, True, False])
best_scores = best_scores.sort_values(by=['dataset_name', 'model_name', 'manifold_type', 'norm_score'], ascending=[True, True, True, False])
# print(best_manifolds.to_string(index=False))
#DEBUG
# best_manifolds = best_manifolds[:9]

fig_rows = best_manifolds['dataset_name'].nunique()
fig_cols = best_manifolds['model_name'].nunique()
# fig, axs = plt.subplots(
#     nrows=fig_rows, 
#     ncols=fig_cols * 2, 
#     figsize=(fig_cols * 5 * 2, fig_rows * 4), 
#     # constrained_layout=Truee
# )
fig, axs = create_custom_subplot_grid(fig_rows, fig_cols, figsize_multiplier=(4, 4))

add_subplot_separators(fig, axs, direction='vertical', every_n=2, margin=(0.1, 0.9))
# add_subplot_separators(fig, axs, direction='horizontal', every_n=1)

# Plot histogram of scores for each dataset
order = ['linear', 'log_linear', 'semicircular', 'log_semicircular', 'circular', 'discrete_circular', 'cluster']

for i, (index, row) in enumerate(best_manifolds.drop(columns=['norm_fold_score', 'fold_score']).drop_duplicates().iterrows()):
    ax_r = i // fig_cols
    ax_c = (i*2+1) % (fig_cols * 2)
    # plt.figure(figsize=(10, 6))
    ax = axs[ax_r][ax_c]  # Use the second column for the bar plot
    subset = best_scores[best_scores['dataset_name'] == row['dataset_name']]
    subset = subset[subset['model_name'] == row['model_name']]
    
    clean_barplot(ax, subset.drop(columns=['norm_fold_score', 'fold_score']).drop_duplicates(), order=order)

    sns.barplot(
        data=subset, 
        x='norm_fold_score', 
        y='manifold', 
        hue='manifold_type',
        errorbar='se',
        order=order,
        palette=reordered_Set1,
        ax=ax,  
    )

    ax.set(xlabel=None)

    # if ax_r == 0 and ax_c == fig_cols*2 - 1:
    #     # First get the existing legend handles and labels
    #     handles, labels = ax.get_legend_handles_labels()

    #     # Desired order
    #     desired_order = ['linear', 'circular', 'cluster']

    #     # Reorder handles and labels according to desired order
    #     sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: desired_order.index(hl[1]))
    #     handles, labels = zip(*sorted_handles_labels)

    #     # Now pass the reordered handles and labels to legend
    #     ax.legend(
    #         handles, labels,
    #         loc='upper right', 
    #         bbox_to_anchor=(2.2, 1), 
    #         title='Manifold Topology',
    #         fontsize=12,
    #         title_fontsize=14,
    #     )

    # else:
    ax.get_legend().set_visible(False)

    # Rescale the y-axis to 0.8 original
    pos = ax.get_position()  # Bbox(x0, y0, x1, y1)
    new_height = pos.height * 0.8
    y_offset = (pos.height - new_height) / 2

    # Set the new position: (x0, y0 + offset, width, new_height)
    ax.set_position([pos.x0, pos.y0 + y_offset, pos.width, new_height])
    ax.tick_params(axis='y', pad=-10, colors='white')
    ax.set_ylabel('')
    # Set alignment and styling for y-tick labels
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_fontweight('bold')
        label.set_color('white')
        # Add white outline
        # label.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
        #                     path_effects.Normal()])

for i, (index, row) in enumerate(best_manifolds.drop(columns=['norm_fold_score', 'fold_score']).drop_duplicates().iterrows()):
    ax_r = i // fig_cols
    ax_c = (i*2) % (fig_cols * 2)
    model_name = row['model_name']
    dataset_name = row['dataset_name']
    layer = row['best_layer']
    manifold = row['manifold']
    manifold_type = row['manifold_type']
    preprocess_func = row['preprocess_func'] if pd.notna(row['preprocess_func']) else None
    label_col = row.get('label_col', None) if pd.notna(row['label_col']) else None  # Use .get() to avoid KeyError if 'label_col' is not present
    postprocess_func = row.get('postprocess_func', None) if pd.notna(row['postprocess_func']) else None  # Use .get() to avoid KeyError if 'postprocess_func' is not present
    clean_scatterplot(axs[ax_r][ax_c])

    print(f"Plotting {model_name} - {dataset_name} - {layer} - {manifold}")
    plot_activation_manifold(model_name, dataset_name, layer, manifold, manifold_type, axs[ax_r][ax_c], target_col=target_col,
                             label_col=label_col, preprocess_func=preprocess_func, postprocess_func=postprocess_func)
    if ax_r == 0:
        # axs[ax_r][ax_c].set_title(model_name, fontsize=14)
        add_shared_title(model_name, axs[ax_r][ax_c], axs[ax_r][ax_c + 1], fig=fig, y_pad=0.02, fontsize=18)
    if ax_c == 0:
        axs[ax_r][ax_c].set_ylabel(dataset_name.replace('_3way',''), fontsize=18)
    # plt.tight_layout()

# fig.show()
fig.savefig('plots/manifolds_scale.pdf', bbox_inches='tight', dpi=300)