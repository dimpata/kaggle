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
from IPython.core.debugger import set_trace


#%%
class CustomImageItemList(ImageItemList):
    def open(self, fn):
        img = fn.reshape(28,28)
        img = np.stack((img,)*3, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))

    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, header:str='infer', **kwargs)->'ItemList':
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        res.items = df.iloc[:,1:].apply(lambda x: x.values, axis=1).values
        return res


#%%
tfms = get_transforms(do_flip=False)
data = (CustomImageItemList.from_csv_custom(path='./data', csv_name='train.csv')
                           .random_split_by_pct(.2)
                           .label_from_df(cols='label')
                           #.transform(tfms)
                           .databunch(bs=64, num_workers=0))
                          


#%%
learn = create_cnn(data, arch=models.resnet34, metrics=accuracy)


#%%
learn.lr_find()
learn.recorder.plot()


#%%
lr = 1e-2
learn.fit_one_cycle(4, lr)


