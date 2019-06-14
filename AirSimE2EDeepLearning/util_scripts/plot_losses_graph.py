import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', '-csv_path', help='path to csv file', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\segmentation_layers_loss\\results_test.csv', type=str)
args = parser.parse_args()

# prepare lists for ploting
steps, loss_4layers, loss_3layers, loss_2layers, loss_1layers = [], [], [], [], []

current_df = pd.read_csv(args.csv_path)
for i in range(0, current_df.shape[0], 1):
    steps.append(int(current_df.iloc[i][['Step']]))
    loss_4layers.append(float(current_df.iloc[i][['4 layers']]))
    loss_3layers.append(float(current_df.iloc[i][['3 layers']]))
    loss_2layers.append(float(current_df.iloc[i][['2 layers']]))
    loss_1layers.append(float(current_df.iloc[i][['1 layers']]))

# plot the results
plt.plot(steps, loss_4layers)
plt.plot(steps, loss_3layers)
plt.plot(steps, loss_2layers)
plt.plot(steps, loss_1layers)
plt.legend(['4 trainable layers','3 trainable layers','2 trainable layers','1 trainable layers'])
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.grid(True)
plt.show()
#plt.savefig(os.path.join(output_dir, 'loss.png'))