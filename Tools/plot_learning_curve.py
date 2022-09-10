import os
import pandas as pd
import matplotlib.pyplot as plt

for folder in os.listdir('.'):
    if not os.path.isdir(folder):
        continue
    file_path = f"./{folder}/log.out"
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        continue
    df = pd.read_csv(file_path)
    if len(df) == 0:
        continue
    df.sort_values(['global step', 'mode'], inplace=True)
    val_df = df[df['mode'] == 'validation']
    val_df.plot(x='global step', y='accuracy')
    # plt
    # plt.scatter(df['global step'], df['accuracy'], linestyle='None')
    plt.ylabel("Val. Accuracy")
    plt.xlabel("Global Steps")
    plt.title(folder)
    plt.savefig(f"{folder}.pdf", format='pdf')
    plt.show()
    print()
