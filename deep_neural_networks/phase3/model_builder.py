from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_model(config, num_classes):
    """
    Builds and compiles the CNN based on WandB configuration.
    """
    model = Sequential([
        Conv2D(75, (3,3), strides=1, padding='same', activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPool2D((2,2), strides=2, padding='same'),
        
        Conv2D(50, (3,3), strides=1, padding='same', activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2,2), strides=2, padding='same'),
        
        Conv2D(25, (3,3), strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2), strides=2, padding='same'),
        
        Flatten(),
        
        # Hyperparameter: Dense Units
        Dense(units=config.dense_units, activation='relu'),
        # Hyperparameter: Dropout
        Dropout(config.dropout),
        
        Dense(units=num_classes, activation='softmax')
    ])

    # Hyperparameter: Optimizer
    if config.optimizer == 'adam':
        opt = Adam(learning_rate=config.learning_rate)
    elif config.optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=config.learning_rate)
    else:
        opt = SGD(learning_rate=config.learning_rate)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model