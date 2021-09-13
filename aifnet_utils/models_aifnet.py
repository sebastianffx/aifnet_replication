import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model_twoPvols(width=256, height=256, num_channels=28):
    """Build a 3D convolutional neural network model."""
    #width and height of the PCT is 256, the number of slices is variable, and the number of channels are
    #the number of timepoints in the PCT sequence        
    inputs = keras.Input((width, height, None , num_channels))

    x = layers.Conv3D(filters=16, kernel_size=(3,3,1), activation="relu", data_format='channels_last', padding='same')(inputs)
    #x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    #x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    #x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=128, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    #x = layers.Conv3D(filters=256, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    #x = layers.Dropout(0.3)(x)
    Lout = layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    Lout2 = layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    P_vol_aif = tf.keras.activations.sigmoid(Lout,name="aif")
    P_vol_vof = tf.keras.activations.sigmoid(Lout2,name="vof")
    #Voxelwise multiplication of P_vol and each of the CTP time points    
    voxelwise_mult_each_ctp_aif = tf.keras.layers.Multiply()([inputs,P_vol_aif])    
    voxelwise_mult_each_ctp_vof = tf.keras.layers.Multiply()([inputs,P_vol_vof])        
    #The 3D average pooling block averages the volumetric information along the x-y-z axes, 
    #such that the predicted vascular function y(t) is a 1D vector of length T.
    outputs_aif = layers.GlobalAveragePooling3D(data_format='channels_last', name='aif_loss')(voxelwise_mult_each_ctp_aif)
    outputs_vof = layers.GlobalAveragePooling3D(data_format='channels_last', name='vof_loss')(voxelwise_mult_each_ctp_vof)    
    #outputs_aif = tf.keras.activations.linear

    # Define the model.
    model = keras.Model(inputs, [outputs_aif,outputs_vof], name="aifnet")
    return model

def get_model_onehead(width=256, height=256, num_channels=43):
    """Build a 3D convolutional neural network model."""
    #width and height of the PCT is 256, the number of slices is variable, and the number of channels are
    #the number of timepoints in the PCT sequence        
    inputs = keras.Input((width, height, None , 43))

    x = layers.Conv3D(filters=16, kernel_size=(3,3,1), activation="relu", data_format='channels_last', padding='same')(inputs)        
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=128, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    #x = layers.Conv3D(filters=256, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    Lout = layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    P_vol_aif = tf.keras.activations.softmax(Lout,name="Pvol_aif")
    
    #Voxelwise multiplication of P_vol and each of the CTP time points    
    voxelwise_mult_each_ctp = tf.keras.layers.Multiply()([inputs,P_vol_aif])    
        
    #The 3D average pooling block averages the volumetric information along the x-y-z axes, 
    #such that the predicted vascular function y(t) is a 1D vector of length T.
    x_aif = layers.GlobalAveragePooling3D(data_format='channels_last')(voxelwise_mult_each_ctp)
    #x = layers.Dense(units=512, activation="relu")(x)
    
    outputs_aif = layers.Dense(units=num_channels, activation="linear",name="aif_pred")(x_aif)

    # Define the model.
    model = keras.Model(inputs, outputs_aif, name="aifnet")
    return model


def get_model_twoheads(width=256, height=256, num_channels=43):
    """Build a 3D convolutional neural network model."""
    #width and height of the PCT is 256, the number of slices is variable, and the number of channels are
    #the number of timepoints in the PCT sequence        
    inputs = keras.Input((width, height, None , 43))

    x = layers.Conv3D(filters=16, kernel_size=(3,3,1), activation="relu", data_format='channels_last', padding='same')(inputs)        
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(filters=128, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    #x = layers.Conv3D(filters=256, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    Lout = layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="relu", data_format='channels_last', padding='same')(x)

    P_vol = tf.keras.activations.softmax(Lout,name="Pvol")
    
    #Voxelwise multiplication of P_vol and each of the CTP time points    
    voxelwise_mult_each_ctp_aif = tf.keras.layers.Multiply()([inputs,P_vol])    
    voxelwise_mult_each_ctp_vof = tf.keras.layers.Multiply()([inputs,P_vol])        
    #The 3D average pooling block averages the volumetric information along the x-y-z axes, 
    #such that the predicted vascular function y(t) is a 1D vector of length T.
    x_aif = layers.GlobalAveragePooling3D(data_format='channels_last')(voxelwise_mult_each_ctp_aif)
    x_vof = layers.GlobalAveragePooling3D(data_format='channels_last')(voxelwise_mult_each_ctp_vof)

    #x = layers.Dense(units=512, activation="relu")(x)
    
    outputs_aif = layers.Dense(units=num_channels, activation="linear",name="aif_pred")(x_aif)
    outputs_vof = layers.Dense(units=num_channels, activation="linear",name="vof_pred")(x_vof)
    
    # Define the model.
    model = keras.Model(inputs, [outputs_aif,outputs_vof], name="aifnet")
    return model