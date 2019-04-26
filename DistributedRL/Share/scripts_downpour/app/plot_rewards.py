import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-batch_size', help='size of the sum of the rewards', default=1000, type=int)
parser.add_argument('--density', '-density', help='configure plot density', default=0.2, type=float)
args = parser.parse_args()

current_df = pd.read_csv(os.path.join(os.getcwd(), 'DistributedRL\\Share\\checkpoint\\local_run\\rewards.txt'), sep='\t')
sums = []
means = []

for i in range(0, current_df.shape[0] - 1): 
    sums.append(float(current_df.iloc[i][['Sum']]))
    means.append(float(current_df.iloc[i][['Mean']]))

# x values for both graphs
x_values = np.arange(0.0, current_df.shape[0] - 1, 1/args.density)

sums_squeezed = []
means_squeezed = []
for i in range(0,len(sums),int(1/args.density)):
    sums_squeezed.append(np.asarray(sums[i:i+int(1/args.density)]).mean())
    means_squeezed.append(np.asarray(means[i:i+int(1/args.density)]).mean())
# y values for sums graph
sums_y_values = np.asarray(sums_squeezed)
means_y_values = np.asarray(means_squeezed)

# plot the results
plt.plot(x_values, sums_y_values)
plt.xlabel('iterations')
plt.ylabel('Sum')
plt.title('Rewards Sum')
plt.grid(True)
plt.show()
plt.plot(x_values, means_y_values)
plt.xlabel('iterations')
plt.ylabel('Mean')
plt.title('Rewards Mean')
plt.grid(True)
plt.show()
#plt.savefig(os.path.basename(args.csv).split(".")[-2]+ ".png")