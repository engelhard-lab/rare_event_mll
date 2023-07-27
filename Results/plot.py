#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns
#%%
data = pd.read_csv('final_sim_scale2.csv', index_col=0)
data["overlap"] = (10-data["n_distinct"])*10
data = data.sort_values(by=['overlap','iter'])
sin = data[data['er2_ratio']==0]
mul = data[data['er2_ratio']>0]

#%% plot1
fig, axs = plt.subplots(2,2, figsize=(8,8))
for i in range(20):
    sim = [0,20,40,60,80,100]
    x = np.linspace(min(sim), max(sim), 300)

    # auc
    single_auc = sin[sin["iter"]==i]["auc"]
    multi_auc = mul[mul["iter"]==i]["auc"]
    curve = make_interp_spline(sim, multi_auc)
    axs[0,0].scatter(sim, multi_auc)
    axs[0,0].plot(x, curve(x))
    color = axs[0,0].get_lines()[-1].get_color()
    axs[0,0].scatter([-20,0],
                     [single_auc.iloc[0],multi_auc.iloc[0]],
                     color=color)
    axs[0,0].plot([-20,0],
                  [single_auc.iloc[0],multi_auc.iloc[0]],
                  linestyle='dashed',
                  color=color)
    axs[0,0].set_xticks([-20,0,20,40,60,80,100])
    axs[0,0].set_xticklabels(['baseline',0,20,40,60,80,100])
    axs[0,0].set_xlabel('overlapping %')
    axs[0,0].set_ylabel('AUC')
    
    # ap
    single_ap = sin[sin["iter"]==i]["ap"]
    multi_ap = mul[mul["iter"]==i]["ap"]
    curve = make_interp_spline(sim, multi_ap)
    axs[0,1].scatter(sim, multi_ap)
    axs[0,1].plot(x, curve(x))
    color = axs[0,1].get_lines()[-1].get_color()
    axs[0,1].scatter([-20,0],
                     [single_ap.iloc[0],multi_ap.iloc[0]],
                     color=color)
    axs[0,1].plot([-20,0],
                  [single_ap.iloc[0],multi_ap.iloc[0]],
                  linestyle='dashed',
                  color=color)
    axs[0,1].set_xticks([-20,0,20,40,60,80,100])
    axs[0,1].set_xticklabels(['baseline',0,20,40,60,80,100])
    axs[0,1].set_xlabel('overlapping %')
    axs[0,1].set_ylabel('AP')

    # r2
    single_r2 = sin[sin["iter"]==i]["r2"]
    multi_r2 = mul[mul["iter"]==i]["r2"]
    curve = make_interp_spline(sim, multi_r2)
    axs[1,0].scatter(sim, multi_r2)
    axs[1,0].plot(x, curve(x))
    color = axs[1,0].get_lines()[-1].get_color()
    axs[1,0].scatter([-20,0],
                     [single_r2.iloc[0],multi_r2.iloc[0]],
                     color=color)
    axs[1,0].plot([-20,0],
                  [single_r2.iloc[0],multi_r2.iloc[0]],
                  linestyle='dashed',
                  color=color)
    axs[1,0].set_xticks([-20,0,20,40,60,80,100])
    axs[1,0].set_xticklabels(['baseline',0,20,40,60,80,100])
    axs[1,0].set_xlabel('overlapping %')
    axs[1,0].set_ylabel('R-square')

    # coe
    single_cov = sin[sin["iter"]==i]["cov"]
    multi_cov = mul[mul["iter"]==i]["cov"]
    curve = make_interp_spline(sim, multi_cov)
    axs[1,1].scatter(sim, multi_cov)
    axs[1,1].plot(x, curve(x))
    color = axs[1,1].get_lines()[-1].get_color()
    axs[1,1].scatter([-20,0],
                     [single_cov.iloc[0],multi_cov.iloc[0]],
                     color=color)
    axs[1,1].plot([-20,0],
                  [single_cov.iloc[0],multi_cov.iloc[0]],
                  linestyle='dashed',
                  color=color)
    axs[1,1].set_xticks([-20,0,20,40,60,80,100])
    axs[1,1].set_xticklabels(['baseline',0,20,40,60,80,100])
    axs[1,1].set_xlabel('overlapping %')
    axs[1,1].set_ylabel('Coefficient')

plt.suptitle('Multi-learning performance by similarity \n \
             er1=0.01, er2=0.05')
plt.tight_layout()
plt.show()

# %% plot2
data_diff = mul.copy()
for i in range(20):
    data_diff.loc[data_diff['iter']==i,'auc'] -= sin.loc[sin['iter']==i,'auc'].squeeze()
    data_diff.loc[data_diff['iter']==i,'ap'] -= sin.loc[sin['iter']==i,'ap'].squeeze()
    data_diff.loc[data_diff['iter']==i,'r2'] -= sin.loc[sin['iter']==i,'r2'].squeeze()
    data_diff.loc[data_diff['iter']==i,'cov'] -= sin.loc[sin['iter']==i,'cov'].squeeze()

id_vars = list(set(data_diff.columns)-
               set(['auc','ap','r2','cov']))
data_diff = pd.melt(data_diff, id_vars=id_vars, var_name='metric_method', value_name='value') 


fig, axs = plt.subplots(2,2, figsize=(8,8))

#auc
data_diff_auc = data_diff[data_diff["metric_method"]=='auc']
sns.lineplot(data=data_diff_auc,
             x='overlap', y='value',
             ax=axs[0,0])
axs[0,0].plot(np.linspace(0, 100, 100),
         np.zeros(100),
         label='Single-label Baseline', linestyle='dashed', color='red')
axs[0,0].legend()
axs[0,0].set_xlabel('overlapping %')
axs[0,0].set_ylabel('AUC')

#ap
data_diff_ap = data_diff[data_diff["metric_method"]=='ap']
sns.lineplot(data=data_diff_ap,
             x='overlap', y='value',
             ax=axs[0,1])
axs[0,1].plot(np.linspace(0, 100, 100),
         np.zeros(100),
         label='Single-label Baseline', linestyle='dashed', color='red')
axs[0,1].legend()
axs[0,1].set_xlabel('overlapping %')
axs[0,1].set_ylabel('AP')

#r2
data_diff_r2 = data_diff[data_diff["metric_method"]=='r2']
sns.lineplot(data=data_diff_r2,
             x='overlap', y='value',
             ax=axs[1,0])
axs[1,0].plot(np.linspace(0, 100, 100),
         np.zeros(100),
         label='Single-label Baseline', linestyle='dashed', color='red')
axs[1,0].legend()
axs[1,0].set_xlabel('overlapping %')
axs[1,0].set_ylabel('R2')

#cov
data_diff_cov = data_diff[data_diff["metric_method"]=='cov']
sns.lineplot(data=data_diff_cov,
             x='overlap', y='value',
             ax=axs[1,1])
axs[1,1].plot(np.linspace(0, 100, 100),
         np.zeros(100),
         label='Single-label Baseline', linestyle='dashed', color='red')
axs[1,1].legend()
axs[1,1].set_xlabel('overlapping %')
axs[1,1].set_ylabel('Coefficient')

plt.suptitle('single-multi performance difference by similarity \n \
             er1=0.01, er2=0.05')
plt.tight_layout()
plt.show()
# %%
