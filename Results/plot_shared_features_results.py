from Results.base_plot_auc_ap import plot_auc_ap

file_name = 'sklearn/shared_features_results.csv'
vars = {
    'x_var': '% overlap',
    'hue_var': 'method',
    'row_var': 'metric',
    'col_var': 'activation'
}
calc_vars = {'% overlap': 'n_overlapping/(n_overlapping+n_distinct)'}

plot_auc_ap(file_name=file_name, vars=vars, calc_vars=calc_vars)
