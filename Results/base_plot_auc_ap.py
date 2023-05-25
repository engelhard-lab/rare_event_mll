import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_auc_ap(file_name, vars, calc_vars=None):
    results = pd.read_csv(file_name)
    for k, v in calc_vars.items():
        results[k] = results.eval(v)
    results = results.drop(
        columns=[c for c in results.columns if c not in vars.values() and not
                 (('auc_' in c) | ('ap_' in c))])
    results = pd.melt(results,
                      id_vars=[c for c in results.columns if
                               not (('auc_' in c) | ('ap_' in c))],
                      value_vars=[c for c in results.columns if
                                  (('auc_' in c) | ('ap_' in c))]
                      )
    results[['metric', 'method']] = results['variable'].str.split('_', n=1,
                                                                  expand=True)
    results = results.drop(columns=['variable'])

    results_plot = sns.catplot(data=results, x=vars['x_var'], y='value',
                               hue=vars['hue_var'],
                               row=vars['row_var'], col=vars['col_var'],
                               kind='box', sharex=True, sharey=False)
    
    for ax in results_plot.axes.flatten():
        ax.set_xticklabels([0.001, 0.003, 0.01, 0.03, 0.1])

    results_plot.figure.savefig(f'{file_name.split(".")[0]}_boxplot.png')
