import sys
sys.path.append("..")

LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM= 0.9

I = 3
I = float(I)

def get_optimizer(parameters, optimizer):
    if optimizer == 'znd':
        from noise_free.znd import ZNDOptimizer
        optimizer = ZNDOptimizer(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'znd_random':
        from random_noise.znd_random import ZNDRandom
        optimizer = ZNDRandom(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'znd_constant':
        from constant_noise.znd_constant import ZNDConstant
        optimizer = ZNDConstant(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'momentum':
        from torch.optim.sgd import SGD
        optimizer = SGD(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'momentum_random':
        from random_noise.momentum_random import MomentumRandom
        optimizer = MomentumRandom(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'momentum_constant':
        from constant_noise.momentum_constant import MomentumConstant
        optimizer = MomentumConstant(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'adam':
        from torch.optim.adam import Adam
        optimizer = Adam(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'adam_random':
        from random_noise.adam_random import AdamRandom
        optimizer = AdamRandom(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'adam_constant':
        from constant_noise.adam_constant import AdamConstant
        optimizer = AdamConstant(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    else:
        print('the optimizer name you have entered is not supported yet')
        sys.exit()

    return optimizer
