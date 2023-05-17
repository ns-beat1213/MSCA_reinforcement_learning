import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


folder_path = "outputs/2way-single-intersection/dqn"  

# dataframe
result_df = pd.DataFrame(columns=["filename", "mean", "variance"])

# read data from DQN agent
for i in range(1, 1001):
    file_name = f"dqn_conn0_ep{i}.csv"
    file_path = os.path.join(folder_path, file_name)
    
    df = pd.read_csv(file_path)
    
    # system_mean_waiting_time
    mean = df["system_mean_waiting_time"].mean()
    variance = df["system_mean_waiting_time"].var()
    
    # add dataframe
    result_df = result_df.append({"filename": file_name, "mean": mean, "variance": variance}, ignore_index=True)
    
# MA
result_df['mean_rolling_10'] = result_df['mean'].rolling(window=10).mean()
result_df['mean_rolling_30'] = result_df['mean'].rolling(window=30).mean()
result_df['mean_rolling_100'] = result_df['mean'].rolling(window=100).mean()


# read data for random policy
folder_path = "outputs/2way-single-intersection/random"  
file_name = "random_conn4_ep1.csv"

file_path = os.path.join(folder_path, file_name)

df = pd.read_csv(file_path)

mean = df["system_mean_waiting_time"].mean()
result_df['baseline'] = mean

# get min for print
min_value = result_df['mean'].min()

# set style
sns.set(style="darkgrid")

# line plot
plt.figure(figsize=(10, 6))
sns.lineplot(y="mean", x=result_df.index, data=result_df, label="Mean Waiting Time")
sns.lineplot(y="baseline", x=result_df.index, data=result_df, label="Baseline Waiting Time")

# title and label
plt.title("Mean Waiting Time in Each Episode", fontsize=18)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Waiting Time (s)", fontsize=14)

# legend
plt.legend(fontsize=12)

# add text
# Add mean and min text to the plot
plt.text(result_df.index[-1] * 0.6, mean, f"Random: {mean:.2f}", fontsize=12, color='blue', backgroundcolor='white')
plt.text(result_df.index[-1] * 0.6, min_value, f"DQN_min: {min_value:.2f}", fontsize=12, color='blue', backgroundcolor='white')


# Save the plot as an image file
plt.savefig("plots/mean_waiting_time_dqn.png", dpi=300, bbox_inches="tight")