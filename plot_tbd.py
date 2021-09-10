import matplotlib.pyplot as plt
# plot noBT
xs = [2.0e-5, 1.0e-5, 9.0e-6, 8.0e-6, 7.5e-6, ]
acc_42 = [12.58, 94.72, 12.4, 98.12, 13.0]
plt.scatter(xs, acc_42, marker='o', label='Seed=42,beam_width=5,train_step=8192')

xs = [1.0e-5, 7.5e-6]
acc_44 = [12.72, 12.62]
plt.scatter(xs, acc_44, marker='v', label='Seed=44,beam_width=5,train_step=8192')

xs = [7.5e-6]
acc_43 = [96.82]
plt.scatter(xs, acc_43, marker='s', label='Seed=43,beam_width=5,train_step=8192')

xs = [2e-5, 1e-5, 5e-6]
acc_42 = [12.88, 12.96, 13.12]
plt.scatter(xs, acc_42, marker='*', label='Seed=42,beam_width=12,train_step<=41xx')

plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("No BackTranslation")
plt.legend()
plt.show()

# plot en+zh
xs = [1.0e-5, 9.0e-6, 8.0e-6, 7.5e-6, ]
acc_42 = [53.04, 93.92, 12.5, 95.5]
plt.scatter(xs, acc_42, marker='o', label='Seed=42,beam_width=5,train_step=8192')

xs = [1.0e-5]
acc_42_bsz8 = [13.04]
plt.scatter(xs, acc_42_bsz8, marker='v', label='Seed=44,beam_width=5,train_step=8192,bsz=8')

xs = [8.0e-6]
acc_43 = [98.12]
plt.scatter(xs, acc_43, marker='s', label='Seed=43,beam_width=5,train_step=8192')

xs = [2e-5, 1e-5, 5e-6]
acc_42 = [13.48, 62.66, 12.46]
plt.scatter(xs, acc_42, marker='*', label='Seed=42,beam_width=12,train_step<=41xx')

plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("En+Zh")
plt.legend()
plt.show()