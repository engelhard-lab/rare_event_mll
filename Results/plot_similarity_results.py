from Results.base_plot_auc_ap import plot_auc_ap

file_name = 'sklearn/similarity_results.csv'
vars = {
    'x_var': 'similarity',
    'hue_var': 'method',
    'row_var': 'metric',
    'col_var': 'activation'
}
calc_vars = {}

plot_auc_ap(file_name=file_name, vars=vars, calc_vars=calc_vars)
