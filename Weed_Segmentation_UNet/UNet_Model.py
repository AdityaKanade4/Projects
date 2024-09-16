# https://youtu.be/jvZm8REF2KY
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers. 
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
from keras.metrics import MeanIoU

def jaccard_coef_multiclass(num_classes):
    def jaccard_coef(y_true, y_pred):
        y_true_f = K.argmax(y_true, axis=-1)
        y_pred_f = K.argmax(y_pred, axis=-1)
        jaccard = 0
        for i in range(num_classes):
            y_true_i = K.cast(K.equal(y_true_f, i), K.floatx())
            y_pred_i = K.cast(K.equal(y_pred_f, i), K.floatx())
            intersection = K.sum(y_true_i * y_pred_i)
            union = K.sum(y_true_i) + K.sum(y_pred_i) - intersection
            jaccard += (intersection + 1.0) / (union + 1.0)
        return jaccard / num_classes
    return jaccard_coef

def dice_coef_multiclass(num_classes):
    def dice_coef(y_true, y_pred):
        y_true_f = K.argmax(y_true, axis=-1)
        y_pred_f = K.argmax(y_pred, axis=-1)
        dice = 0
        for i in range(num_classes):
            y_true_i = K.cast(K.equal(y_true_f, i), K.floatx())
            y_pred_i = K.cast(K.equal(y_pred_f, i), K.floatx())
            intersection = K.sum(y_true_i * y_pred_i)
            dice += (2. * intersection + 1.0) / (K.sum(y_true_i) + K.sum(y_pred_i) + 1.0)
        return dice / num_classes
    return dice_coef

def mean_iou_multiclass(num_classes):
    def mean_iou(y_true, y_pred):
        y_true_f = K.argmax(y_true, axis=-1)
        y_pred_f = K.argmax(y_pred, axis=-1)
        iou = 0
        for i in range(num_classes):
            y_true_i = K.cast(K.equal(y_true_f, i), K.floatx())
            y_pred_i = K.cast(K.equal(y_pred_f, i), K.floatx())
            intersection = K.sum(y_true_i * y_pred_i)
            union = K.sum(y_true_i) + K.sum(y_pred_i) - intersection
            iou += (intersection + 1.0) / (union + 1.0)
        return iou / num_classes
    return mean_iou



################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model
 