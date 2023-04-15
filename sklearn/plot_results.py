import pandas as pd
import seaborn as sns

results = pd.read_csv('results/results.csv')
results = results.drop(columns=['n', 'p', 'er'])

results = pd.melt(results, id_vars=['activation', 'sim', 'iter'])
results[['metric', 'method']] = results['variable'].str.split('_', expand=True)
results = results.drop(columns=['variable'])

results_plot = sns.catplot(data=results, x='sim', y='value', hue='method',
                           row='metric', col='activation', kind='box',
                           sharex=True, sharey=False)

results_plot.figure.savefig('results/results_boxplot.png')
