import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from main import  df_model_bogo

temp_df = df_model_bogo.mean()
temp_df = temp_df.reset_index(drop=False)
temp_df.columns = ['Features', 'Mean']
temp_df = temp_df.iloc[2:,:]
# temp_df = temp_df.sort_values('Mean', ascending=False)

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(5, 3), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.4, hspace=0.1)

background_color = "#f6f5f5"
sns.set_palette(['#ffa600']*11)

ax = fig.add_subplot(gs[0, 0])
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.set_facecolor(background_color)
ax_sns = sns.barplot(ax=ax, x=temp_df['Features'],
                      y=temp_df['Mean'],
                      zorder=2, linewidth=0, alpha=1, saturation=1)
ax_sns.set_xlabel("Features",fontsize=4, weight='bold')
ax_sns.set_ylabel("Mean",fontsize=4, weight='bold')
ax_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.tick_params(labelsize=2.5, width=0.5, length=1.5)
ax.text(-0.5, 0.8, 'Mean Values - BOGO Treatment', fontsize=6, ha='left', va='top', weight='bold')
ax.text(-0.5, 0.7, 'Each feature get value of 0 (No) and 1 (yes) => Mean is the % of Yes', fontsize=4, ha='left', va='top')
# data label
for p in ax.patches:
    percentage = f'{p.get_height():.1f}' ##{:. 0%}
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.05
    ax.text(x, y, percentage, ha='center', va='center', fontsize=3,
           bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.3))

# 调整顶部边距
plt.subplots_adjust(top=0.85)

plt.show()
