import pandas as pd
import os
import numpy as np
from matplotlib import *
from matplotlib.pyplot import *

train_file = sys.argv[1]
test_file = sys.argv[2]

train_log = pd.read_csv(os.path.join('../logs/split/', train_file))
test_log = pd.read_csv(os.path.join('../logs/split/', test_file))

mean = np.mean(train_log["loss"])
ylim1 = (mean-(mean*3),mean+(mean*4.5))
mean = np.mean(test_log["loss"])
ylim2 = (mean-(mean*.4),mean+(mean*.4))

_, ax1 = subplots(figsize=(15, 10))
ylim(ylim1)
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ylim(ylim2)
ax2.plot(test_log["NumIters"], test_log["loss"], 'g')

ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')

trmin = train_log["loss"].idxmin()
temin = test_log["loss"].idxmin()

text = 'Training loss min: '+str(train_log.iloc[[trmin]].loss.item())+' at iteration: '+str(train_log.iloc[[trmin]].NumIters.item())+\
'\nTest loss min: '+str(test_log.iloc[[temin]].loss.item())+' at iteration: '+str(test_log.iloc[[temin]].NumIters.item())

ann = (np.mean(train_log["NumIters"]),mean+(mean*.2))

ax2.annotate(text, xy=ann,)
ax2.set_ylabel('test loss')

name = train_file.split('.')[0]

savefig("../figures/"+name+"_train_test_image.png") #save image as png
