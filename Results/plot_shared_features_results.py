from Results.base_plot_auc_ap import plot_auc_ap

file_name = 'torch/raytune.csv'

# vars = {
#     'x_var': 'er',
#     'hue_var': 'method',
#     'row_var': 'metric',
#     'col_var': 'n_distinct'
# }
# calc_vars = {}

# vars = {
#     'x_var': 'hidden_layers',
#     'hue_var': 'method',
#     'row_var': 'metric',
#     'col_var': 'activation'
# }
# calc_vars = {}

# vars = {
#     'x_var': '% overlap',
#     'hue_var': 'method',
#     'row_var': 'metric',
#     'col_var': 'activation'
# }

vars = {
    'x_var': '% overlap',
    'hue_var': 'metric',
    'row_var': 'metric',
    'col_var': 'er'
}

calc_vars = {'% overlap': '(1-n_distinct/n_random_features)',
             'diff_auc': '(auc_multi-auc_single)',
             'diff_ap': '(ap_multi-ap_single)'}

plot_auc_ap(file_name=file_name, vars=vars, calc_vars=calc_vars)
