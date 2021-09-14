import numpy as np
import matplotlib.pyplot as plt
from aifnet_utils.preprocess import read_nifti_file, normalize, normalize_aif, process_scan, normalize_zero_one
import keras.backend as K
import gc 

def plot_predictions(model,x,y,prefix_fig, normalize_preds=True, type_pred='AIF', savefig=True):
    pred = model.predict(np.expand_dims(normalize(x), axis=0))    
    if type_pred=='AIF':
        pred = pred[0][0]
        plt.title('AIF Function Predictions')        
    else:
        pred = pred[1][0]
        plt.title('VOF Function Predictions')

    if normalize_preds:
        #print('Normalizing')
        pred = normalize_zero_one(pred)
        y = normalize_zero_one(y)
    #plt.plot(y)
    #plt.plot(pred)    
    #plt.legend(['y', 'prediction'])
    #plt.xlabel('Time (s)')
    #plt.ylabel('Normalized Density (HU)')
    if savefig:
        #plt.savefig(prefix_fig + '.png')
        #plt.figure().clear()
        #plt.close()
        #plt.cla()
        #plt.clf()
        gc.collect()
        K.clear_session()
        return y,pred

    else:
        plt.show()
    return