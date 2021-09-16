import numpy as np
import matplotlib.pyplot as plt
from aifnet_utils.preprocess import read_nifti_file, normalize, normalize_aif, process_scan, normalize_zero_one
#import keras.backend as K
import gc 

def plot_predictions(model,x,y,prefix_fig, normalize_preds=True, type_pred='AIF', savefig=True):
    pred = model.predict(np.expand_dims(normalize(x), axis=0))    
    if normalize_preds:
    #print('Normalizing')
        pred[0][0] = normalize_zero_one(pred[0][0])
        pred[1][0] = normalize_zero_one(pred[1][0])
        y = normalize_zero_one(y)

    if type_pred=='AIF':
        pred = pred[0][0]
        plt.title('AIF Function Predictions')        
        plt.plot(y)
        plt.plot(pred)    
        plt.legend(['y', 'prediction'])
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Density (HU)')

    if type_pred=='VOF':
        pred = pred[1][0]
        plt.title('VOF Function Predictions')
        plt.plot(y)
        plt.plot(pred)    
        plt.legend(['y', 'prediction'])
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Density (HU)')

    if type_pred=='BOTH':
        predictions = [pred[0][0], pred[1][0]]
        plt.title('AIF - VOF Function Predictions')
        plt.plot(y[0])
        plt.plot(y[1])
        plt.plot(predictions[0])
        plt.plot(predictions[1])
    if savefig:
        #print("Saving prediction for case: " + str(prefix_fig.split('/')[-1].split('_')[-1])) 
        plt.savefig(prefix_fig + '.png')
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        #gc.collect()
        #K.clear_session()
        return y,pred

    else:
        plt.show()
    return