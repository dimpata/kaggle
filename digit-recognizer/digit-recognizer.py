#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'digit-recognizer'))
	print(os.getcwd())
except:
	pass


#%%
from fastai import *
from fastai.vision import *


#%%
class CustomImageItemList(ImageItemList):
    def open(self, fn):
        tensor = torch.load(fn)
        return Image(tensor)


#%%
path = f'./data'
train_df = pd.read_csv(f'{path}/train.csv')
train_df.sample(5)


