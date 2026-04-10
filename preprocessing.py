# Load the dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("cloud_data.csv")
print("Shape of dataset: ",df.shape)
print("\nColumns: ",df.columns)


# Define thresholds (using quantiles for dynamic scaling)
cpu_thresh = df['cpu_usage'].quantile(0.2)
mem_thresh = df['memory_usage'].quantile(0.2)
net_thresh = df['network_traffic'].quantile(0.2)
ins_thresh = df['num_executed_instructions'].quantile(0.2)
eff_thresh = df['energy_efficiency'].quantile(0.2)

# Create conditions (each gives 1 if true, 0 if false)
conditions = (
    (df['cpu_usage'] <= cpu_thresh).astype(int) +
    (df['memory_usage'] <= mem_thresh).astype(int) +
    (df['network_traffic'] <= net_thresh).astype(int) +
    (df['num_executed_instructions'] <= ins_thresh).astype(int) +
    (df['energy_efficiency'] <= eff_thresh).astype(int)
)



# Final zombie label (at least 3 conditions true)
df['zombie'] = (conditions >= 3).astype(int)
df['zombie'].value_counts()
print('\nDataset features before timestamp conversion :\n',df.columns.tolist())

# Step 1: Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 2: Extract features
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday

# Step 3: Now drop columns
df = df.drop(['vm_id','timestamp','task_status'], axis=1)
print('\nDataset features after timestamp conversion :\n',df.columns.tolist())

#check missing values.
print('\nNumber of missing values : \n',df.isnull().sum())



#handle missing values.
#for numeric columns.
numeric_cols=df.select_dtypes(include=['float64','int64']).columns
df[numeric_cols]=df[numeric_cols].fillna(df[numeric_cols].median())

#for categorical columns.
categorical_cols=df.select_dtypes(include=['object']).columns
df[categorical_cols]=df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

#verify everything is clean.
print('\nVerify missing values are handled : \n',df.isnull().sum())



#data validation.
#1 - statistical overview
print('\n',df.describe())

#2 - check column vice.
print("\n\nData validation column vice : ")
print("\nCPU > 100: ",(df['cpu_usage']>100).sum())
print("CPU < 0: ",(df['cpu_usage']<0).sum())
print("Negative power: ",(df['power_consumption']<0).sum())
print("Negative network traffic: ",(df['network_traffic']<0).sum())
print("Negative power:", (df['power_consumption'] < 0).sum())
print("Negative instructions:", (df['num_executed_instructions'] < 0).sum())
print("Negative execution time:", (df['execution_time'] < 0).sum())
print("Negative efficiency:", (df['energy_efficiency'] < 0).sum())



#outlier detection. [IQR method]

num_cols = ['cpu_usage','memory_usage','network_traffic','power_consumption','num_executed_instructions','execution_time','energy_efficiency']
for col in num_cols:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3-Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[col] = df[col].clip(lower_bound,upper_bound)

print('\nOutlier Detection using IQR Method : ',df.describe())




#Exploratory Data Analysis [EDA] -

numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
print('\n\nExploratory Data Analysis\n')
print('Correlation Heatmap - \n')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(
    data=df,
    x='cpu_usage',
    hue='zombie',
    bins=50,
    kde=True,
    stat='density',
    common_norm=False,
    palette={0: 'green', 1: 'red'}
)

plt.legend(title="Server Status", labels=["Zombie Server ", "Active Server"])
plt.title("CPU Usage Distribution: Active vs Zombie Servers")
plt.xlabel("CPU Usage")
plt.ylabel("Density")
print('\n\n')
print('CPU Usage Histogram of Active and Zombie Servers - \n')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(
    data=df,
    x='network_traffic',
    hue='zombie',
    bins=50,
    kde=True,
    stat='density',
    common_norm=False,
    palette={0: 'green', 1: 'red'}
)

plt.legend(title="Server Status", labels=["Zombie Server", "Active Server"])
plt.title("Network Traffic Distribution: Active vs Zombie Servers")
plt.xlabel("Network Traffic")
plt.ylabel("Density")
print('\n\n')
print('Network Traffic Histogram of Active and Zombie Servers - \n')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='zombie', y='cpu_usage', data=df)
plt.xlabel("Server Status")
plt.ylabel("CPU Usage (%)")
plt.xticks([0, 1], ["Active Server", "Zombie Server"])
print('\n\n')
print('Server Status Boxplot for Active and Zombie Servers - \n')
plt.show()