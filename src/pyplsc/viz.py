
from matplotlib import pyplot as plt
import numpy as np

from pdb import set_trace

def plot_boot_stat(df, mapping, ax=None, ylabel=None,
                   bar_width=0.8,
                   group_gap=0.1,
                   capsize=None):
    """
    Create barplot of bootstrap statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of the ``get_boot_stat_frame`` method.
    mapping : iterable or dict
        Dictionary mapping from aesthetics (``'x'``, ``'hue'``, ``'row'``, or ``'column'``)  to variables, or else an iterable of the variables to be plotted.
    ax : matplotlib.axes.Axes, optional
        Axes to plot to. The default is None, in which case a new figure is created.
    ylabel : str, optional
        Name of statistic being plotted. The default is None.
    bar_width : float, optional
        Width of bars. The default is 0.8.
    group_gap : float, optional
        Gap between different groups of bars within the same axis. The default is 0.1.
    capsize : float, optional
        Cap size for confidence intervals. The default is None.

    Returns
    -------
    f : matplotlib.figure.Figure
        Figure containing the plot.

    Examples
    --------
    >>> f, ax = plt.subplots()
    >>> df = mod.get_boot_stat_frame(lv_idx=0, ci='len')
    >>> plot_bot_stat(df, mapping={'hue': 'covariate'}, ax=ax)
    
    """
    n_strat = len(mapping)
    if n_strat > 4:
        raise ValueError('Cannot create a plot for more than 4 stratifying variables')
    auto_aes = ['x', 'hue', 'row', 'column']
    auto_mapping = {k: None for k in auto_aes}
    if isinstance(mapping, dict):
        for k in mapping:
            auto_mapping[k] = mapping[k]
    else:
        mapping = list(mapping)
        for i in range(n_strat):
            aes = auto_aes[i]
            var = mapping[::-1][i]
            auto_mapping[aes] = var
    mapping = auto_mapping
    
    # Count number of levels for each variable
    mapping_counts = {}
    mapping_levels = {}
    for aes, var in mapping.items():
        if var is None:
            mapping_levels[aes] = [None]
            mapping_counts[aes] = 1
        else:
            levels = df[var].unique()
            mapping_levels[aes] = levels
            mapping_counts[aes] = len(levels)
    
    # Get subdivided axes
    nrows = mapping_counts['row']
    ncols = mapping_counts['column']
    f, ax = _get_boot_stat_ax(ax, nrows, ncols)
    
    # Boot done?
    has_yerr = 'L_CI' in df and 'U_CI' in df
    
    # Set up bar positions and hues
    nhue = mapping_counts['hue']
    slot_width = bar_width / max(nhue, 1)
    x_base = np.arange(mapping_counts['hue']) * (1 + group_gap)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {h: cycle[i % len(cycle)] for i, h in enumerate(mapping_levels['hue'])}
    
    # Generate plots
    legend_handles = {}
    for row_idx, row_level in enumerate(mapping_levels['row']):
        for col_idx, col_level in enumerate(mapping_levels['column']):
            curr_ax = ax[row_idx, col_idx]
            rc_df = df
            # Subset for row and column variables
            row_var = mapping['row']
            if row_var is not None:
                rc_df = rc_df[rc_df[row_var] == row_level]
            col_var = mapping['column']
            if col_var is not None:
                rc_df = rc_df[rc_df[col_var] == col_level]
            
            for hue_idx, hue_level in enumerate(mapping_levels['hue']):
                h_df = rc_df
                # Subset for hue
                hue_var = mapping['hue']
                if hue_var is not None:
                    h_df = h_df[h_df[hue_var] == hue_level]

                x_var = mapping['x']
                if x_var is not None:
                    h_df = h_df.set_index(x_var).reindex(mapping_levels['x'])
                else:
                    h_df = h_df.reset_index(drop=True)

                heights = h_df['stat'].values
                if has_yerr:
                    yerr = h_df[['L_CI', 'U_CI']].to_numpy().T

                offset = (hue_idx - (nhue - 1) / 2) * slot_width
                positions = x_base + offset

                bars = curr_ax.bar(
                    positions,
                    heights,
                    width=slot_width * 0.95,
                    yerr=yerr if has_yerr else None,
                    capsize=capsize,
                    color=color_map.get(hue_level),
                    label=hue_level,
                    error_kw=dict(lw=1),
                )
                if mapping['hue'] is not None and hue_level not in legend_handles:
                    legend_handles[hue_level] = bars[0]

            curr_ax.set_xticks(x_base)
            if mapping['x'] is not None:
                curr_ax.set_xticklabels([str(v) for v in mapping_levels['x']])
            
            # Label row and column
            ax_title_parts = []
            if row_var is not None:
                ax_title_parts.append('%s=%s' % (row_var, row_level))
            if col_var is not None:
                ax_title_parts.append('%s=%s' % (col_var, col_level))
            if len(ax_title_parts) > 0:
                curr_ax.set_title(", ".join(ax_title_parts))
            
            # Add x/y labels only as applicable
            if col_idx == 0:
                curr_ax.set_ylabel(ylabel)
            if row_idx == nrows - 1:
                curr_ax.set_xlabel(mapping['x'])

    hue_var = mapping['hue']
    if hue_var is not None and legend_handles:
        f.legend(
            legend_handles.values(),
            [str(k) for k in legend_handles.keys()],
            title=hue_var,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )

    return f

def _get_boot_stat_ax(ax, nrows, ncols):
    if ax is None:
        f, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    else:
        f = ax.get_figure()
        subplotspec = ax.get_subplotspec()
        ax.remove()
        
        gs = subplotspec.subgridspec(nrows, ncols, hspace=0.35, wspace=0.15)

        ax = np.empty((nrows, ncols), dtype=object)
        first_ax = None
        for i in range(nrows):
            for j in range(ncols):
                share_kwargs = {}
                if first_ax is not None:
                    share_kwargs["sharex"] = first_ax
                    share_kwargs["sharey"] = first_ax
                new_ax = f.add_subplot(gs[i, j], **share_kwargs)
                ax[i, j] = new_ax
                if first_ax is None:
                    first_ax = new_ax
    
    return f, ax
