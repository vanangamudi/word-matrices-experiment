import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'imdbreviews'
    dataset_path = '../dataset/aclimdb'
    trainset_size = 1
    hidden_dim = 500
    embed_dim = 100
    num_layers = 1
    
    LR = 0.01
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
