from Results.base_plot_auc_ap import plot_auc_ap

file_name = 'sklearn/test_rs.csv'
vars = {
    'x_var': 'hidden_layers',
    'hue_var': 'method',
    'row_var': 'metric',
    'col_var': 'activation'
}
calc_vars = {'% overlap': '(1-n_distinct/n_random_features)'}

plot_auc_ap(file_name=file_name, vars=vars, calc_vars=calc_vars)
