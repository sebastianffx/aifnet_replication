import numpy as np
import matplotlib.pyplot as plt
from aifnet_utils.preprocess import read_nifti_file, normalize_single_volume, normalize_aif, process_scan, normalize_zero_one, normalize_volumes_in_sequence
#import keras.backend as K
import gc 

def pearson_corr(y,y_hat):
    pearson_numerator = np.sum((y-np.average(y))*(y_hat-np.average(y_hat)))
    pearson_denominator = np.sqrt(np.sum(np.square(y-np.average(y)))) *np.sqrt(np.sum(np.square(y_hat-np.average(y_hat))))
    pearson_correlation = np.abs(pearson_numerator/pearson_denominator)
    return pearson_correlation

def plot_predictions(model,x,y,prefix_fig, normalize_preds=True, type_pred='AIF', savefig=True):
    pred = model.predict(np.expand_dims(normalize(x), axis=0))   
    shape_preds = np.array(pred).shape
    if normalize_preds and type_pred =='BOTH':
        pred[0][0] = normalize_zero_one(pred[0][0])
        pred[1][0] = normalize_zero_one(pred[1][0])
        y = normalize_zero_one(y)
    if normalize_preds and type_pred !='BOTH':
        y = normalize_zero_one(y)
        pred = normalize_zero_one(pred[0])
    if savefig:
        if type_pred=='AIF':
            #pred = pred[0][0]
            #print(pred.shape)
            #print(y.shape)
            #print(pred[0].shape)
            pearson_correlation = pearson_corr(y,pred)
            plt.title('AIF Function Predictions')        
            plt.plot(y,'g-')
            #plt.plot(pred,'w-')
            plt.plot(pred,'b-')    
            #plt.legend(['y', 'corr='+str(pearson_correlation)[0:6] , 'prediction'])
            plt.legend(['y' , 'prediction'])
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized Density (HU)')

        if type_pred=='VOF':
            #pred = pred[1][0]
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
            #print("Saving prediction for case: " + str(prefix_fig.split('/')[-1].split('_')[-1]))             
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
            #gc.collect()
            #K.clear_session()
        plt.savefig(prefix_fig + '.png', dpi=300)
    return (y,pred)