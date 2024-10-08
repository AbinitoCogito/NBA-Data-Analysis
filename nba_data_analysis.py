import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option('display.max_columns', None)

df = pd.read_csv('all_seasons.csv')


# Descriptive statistics 
df.head()
df.info()
df.describe().round(2)
df.shape
df.columns


# Dropping the unnamed column
df.drop(columns='Unnamed: 0',inplace = True)


# Filtered most known players 
kd_filtered = df[df['player_name'] == 'Kevin Durant']
kobe_filtered = df[df['player_name'] == 'Kobe Bryant']
lb_filtered = df[df['player_name'] == 'LeBron James']
je_filtered = df[df['player_name'] == 'Joel Embiid']

print(f' Kevin Durant :\n {kd_filtered}')
print(f' Kobe Bryant :\n {kobe_filtered}')
print(f' LeBron James :\n {lb_filtered}')
print(f' Joel Embiid :\n {je_filtered}')



# Numerical and categorical variables
numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_vars = df.select_dtypes(include=['object']).columns.tolist()                           

print('Numerical variables:', numerical_vars)
print('Categorical variables:', categorical_vars)


for i, col in enumerate(numerical_vars):
    sns.distplot(df[col], hist=True, kde=True, bins=20)
    plt.show()



# Data Preprocessing

# Missing value analysis
missing_columns = [colname for colname in df.columns if df[colname].isna().any()]
print(f'Missing values happens to be in : {missing_columns}', end='\n\n')

missing_columns_dist = df.isna().sum()
print('Missing values distribution within all columns:')
print(missing_columns_dist, end='\n\n')

missing_total = df.isna().sum().sum()
print(f'Total missing values: {missing_total}')

missing_ratio = missing_total / df.size
print(f'Ratio of missing values: {missing_ratio}')




# Dividing the dataset and creating a DataFrame each for East and West Conferences
nba_teams = df['team_abbreviation'].unique()
print(nba_teams)
len(nba_teams)

east_conf = [team for team in nba_teams if team in {'WAS', 'ORL', 'MIL', 'DET', 'BOS', 'IND', 'MIA', 'ATL','NJN', 'PHI', 'NYK', 'TOR', 'CHI', 'CLE', 'CHA', 'BKN'}]
west_conf = [team for team in nba_teams if team in {'HOU', 'VAN', 'LAL', 'DEN', 'CHH', 'POR', 'DAL', 'UTA','SEA', 'SAS', 'LAC', 'GSW', 'PHX', 'MIN', 'SAC', 'MEM','NOH', 'NOK', 'OKC', 'NOP'}]

print(f' East conference teams : {east_conf}')
print(f' West conference teams : {west_conf}')


df_east_conf = df[df['team_abbreviation'].isin(east_conf)]
df_west_conf = df[df['team_abbreviation'].isin(west_conf)]

print(f' East Conference Teams DataFrame :\n {df_east_conf}')
print(f' West Conference Teams DataFrame :\n {df_west_conf}')



# Filtering East and west conferences for teams and colleges
east_team = df_east_conf.groupby('team_abbreviation')
east_team.first()

west_team = df_west_conf.groupby('team_abbreviation')
west_team.first()


east_college = df_east_conf.groupby('college')
east_college.first()

west_college = df_west_conf.groupby('college')
west_college.first()



# Comparing for averages' average statistics for both conferences
east_mean_pts = df_east_conf['pts'].mean().round(2)
print(f' East Conference teams average point :\n {east_mean_pts}')
west_mean_pts = df_west_conf['pts'].mean().round(2)
print(f' West Conference teams average point :\n {west_mean_pts}')


east_mean_ast = df_east_conf['ast'].mean().round(2)
print(f' East Conference teams average assist :\n {east_mean_ast}')
west_mean_ast = df_west_conf['ast'].mean().round(2)
print(f' West Conference teams average assist :\n {west_mean_ast}')


east_mean_reb = df_east_conf['reb'].mean().round(2)
print(f' East Conference teams average rebound :\n {east_mean_reb}')
west_mean_reb = df_west_conf['reb'].mean().round(2)
print(f' West Conference teams average rebound :\n {west_mean_reb}')




# Dropping the missing value column and and the string columns
def fe(df, drop):
    df = df.drop(drop, axis =1)
    

    return df

drop = ['player_name', 'team_abbreviation', 'college','draft_round','draft_number','draft_year', 'country']
nba = fe(df, drop)
nba.info()
nba.head()


# Feature engineering
nba['player_height_m'] = nba['player_height'] / 100
nba['BMI'] = nba['player_weight'] / (nba['player_height_m'] ** 2)
print(nba['BMI'])


plt.title('BMI Distribution Plot')
sns.distplot(nba['BMI'])
plt.show()



# Converting the season column from string to integer
nba['season'] = nba['season'].apply(lambda x: int(x[:4]))

# For example filtering the season 96' players
nba[nba['season'] == 1996]


#Dividing the dataset into 3 groups that have deeply influenced the history of NBA and understanding of the game

# NBA 1996-2004 Isolation and Mid-Range Dominance 
nba_ISO = nba[nba['season'] <= 2004]

# NBA 2004-2015 Defensive Strategy and Analytics
nba_DSA = nba[(nba['season'] >= 2005) & (nba['season'] < 2015)]

# NBA 2015-2022 Three-Point Revolution and Positionless Basketball
nba_TPB = nba[(nba['season'] >= 2015)]



# Exploratory Data Analysis


# Univariate Analysis
# Histogram plots
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(nba_ISO['player_height'], bins=40, label='Player Height')
axs[0].set_xlabel('Height')
axs[1].hist(nba_ISO['player_weight'], bins=40, color='green',label='Player Weight')
axs[1].set_xlabel('Weight')
plt.suptitle(t='Between 1996-2004 Seasons Height And Weight Distributions')
plt.show()


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(nba_DSA['player_height'], bins=40, label='Player Height')
axs[0].set_xlabel('Height')
axs[1].hist(nba_DSA['player_weight'], bins=40, color='green',label='Player Weight')
axs[1].set_xlabel('Weight')
plt.suptitle(t='Between 2004-2015 Seasons Height And Weight Distributions')
plt.show()


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(nba_TPB['player_height'], bins=40, label='Player Height')
axs[0].set_xlabel('Height')
axs[1].hist(nba_TPB['player_weight'], bins=40, color='green',label='Player Weight')
axs[1].set_xlabel('Weight')
plt.suptitle(t='Between 2015-2022 Seasons Height And Weight Distributions')
plt.show()



#Bivariate Analysis
# Box plot through the seasons 
plt.figure(figsize=(14,6))
sns.boxplot(data=nba_ISO, x="season", y="ts_pct")
plt.show()


plt.figure(figsize=(14,6))
sns.boxplot(data=nba_DSA, x="season", y="ts_pct")
plt.show()


plt.figure(figsize=(14,6))
sns.boxplot(data=nba_TPB, x="season", y="ts_pct")
plt.show()




# Multivariate Analysis
# Correlation matrix heatmaps of NBA 1996-2004
sns.heatmap(nba_ISO.corr(),annot=True) 
plt.show()

# Correlation heatmaps of NBA 2004-2015
sns.heatmap(nba_DSA.corr(),annot=True) 
plt.show()


# Correlation heatmap of NBA 2015-2022
sns.heatmap(nba_TPB.corr(),annot=True) 
plt.show()


# Correlation upper triangle heatmaps
mask_ISO = np.triu(np.ones_like(nba_ISO.corr(), dtype=int))
heatmap = sns.heatmap(nba_ISO.corr(), mask=mask_ISO, vmin=-1, vmax=1, annot=False, cmap='BrBG')
plt.show()

mask_DSA = np.triu(np.ones_like(nba_DSA.corr(), dtype=int))
heatmap = sns.heatmap(nba_DSA.corr(), mask=mask_DSA, vmin=-1, vmax=1, annot=False, cmap='BrBG')
plt.show()

mask_TPB = np.triu(np.ones_like(nba_TPB.corr(), dtype=int))
heatmap = sns.heatmap(nba_TPB.corr(), mask=mask_TPB, vmin=-1, vmax=1, annot=False, cmap='BrBG')
plt.show()

