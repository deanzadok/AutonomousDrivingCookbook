import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', '-csv_path', help='path to csv file', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\depth_estimation_layers_loss\\results_test.csv', type=str)
args = parser.parse_args()

# prepare lists for ploting
steps = [4,3,2,1]
loss_10epochs, loss_20epochs, loss_30epochs, loss_40epochs = [], [], [], []
columns = ['4 layers','3 layers','2 layers','1 layers']

current_df = pd.read_csv(args.csv_path)
for i in range(0, len(columns), 1):
    loss_10epochs.append(float(current_df.iloc[9][[columns[i]]]))
    loss_20epochs.append(float(current_df.iloc[19][[columns[i]]]))
    loss_30epochs.append(float(current_df.iloc[29][[columns[i]]]))
    loss_40epochs.append(float(current_df.iloc[39][[columns[i]]]))

# plot the results
plt.plot(steps, loss_10epochs)
plt.plot(steps, loss_20epochs)
plt.plot(steps, loss_30epochs)
plt.plot(steps, loss_40epochs)
plt.xticks([1,2,3,4])
plt.legend(['10 epochs','20 epochs','30 epochs','40 epochs'])
plt.xlabel('Trainable layers')
plt.ylabel('Loss')
plt.title('Loss per trainable layers')
plt.grid(True)
plt.show()
#plt.savefig(os.path.join(output_dir, 'loss.png'))