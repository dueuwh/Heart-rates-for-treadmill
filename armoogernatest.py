import os
import matplotlib.pyplot as plt
import json

model_losseslog_dir = "D:/home/BCML/drax/PAPER/TimeMoE/Time-MoE-50M_Fine-Tuning/model_save/treadmill/loss_hitory.json"
with open(model_losseslog_dir, 'r') as f:
    json_data = json.load(f)

print(max(json_data['train_loss'])+10)

fig = plt.figure(figsize=(10,8))
fig.set_facecolor('white')
plt.plot(json_data['train_loss'])
current_values = plt.gca().get_yticks()
plt.title("train loss log")
plt.gca().set_yticklabels(['{:.4f}'.format(x) for x in current_values])
 
plt.show()

plt.plot(json_data['test_loss'])
plt.title("test loss log")
plt.show()
