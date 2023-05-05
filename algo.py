import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

# Load the clinical data


data = pd.read_csv('/Users/mainuser/Desktop/data/clinical_data.csv')


# Compare the means of two groups using a t-test
group1 = data.loc[data['group'] == 1, 'value']
group2 = data.loc[data['group'] == 2, 'value']
t_stat, p_value = ttest_ind(group1, group2)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Visualize the distribution of a variable by group
sns.boxplot(x='group', y='value', data=data)
plt.show()
