import jittor as jt
from jittor import nn, Module

from models.generator.frame_generator import Generator
from models.discriminators.identity_discriminator import IdentityDiscriminator
from models.discriminators.sequence_discriminator import SequenceDiscriminator

from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform

batch_size = 50

G = Generator()
G_optim = G.get_optimizer()
D_id = IdentityDiscriminator()
D_id_optim = D_id.get_optimizer()
D_se = SequenceDiscriminator()
D_se_optim = D_se.get_optimizer()

transform = transform.Compose([
    transform.Resize(size=[200,100]),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dir = './dataset/train'
train_loader = ImageFolder(train_dir).set_attrs(transform=transform, batch_size=batch_size, shuffle=True)

def ls_loss(x, b):
    mini_batch = x.shape[0]
    y_real_ = jt.ones((mini_batch,))
    y_fake_ = jt.zeros((mini_batch,))
    if b:
        return (x-y_real_).sqr().mean()
    else:
        return (x-y_fake_).sqr().mean()

def train(epoch):
    for batch_idx, (x_, target) in enumerate(train_loader):
        mini_batch = x_.shape[0]

        # train discriminator
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_id_result = D_id(x_, x_)
        D_id_real_loss = ls_loss(D_id_result, True)
        D_id_result_ = D_id(G_result,x_)
        D_id_fake_loss = ls_loss(D_id_result_, False)
        D_id_train_loss = D_id_real_loss + D_id_fake_loss
        D_id_train_loss.sync()
        D_id_optim.step(D_id_train_loss)

        D_se_result = D_se(x_)
        D_se_real_loss = ls_loss(D_se_result, True)
        D_se_result_ = D_se(G_result)
        D_se_fake_loss = ls_loss(D_se_result_, False)
        D_se_train_loss = D_se_real_loss + D_se_fake_loss
        D_se_train_loss.sync()
        D_se_optim.step(D_se_train_loss)

        # train generator
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_id_result = D_id(G_result)
        D_se_result = D_se(G_result)
        G_train_loss = ls_loss(D_id_result, True) + ls_loss(D_se_result, True)
        G_train_loss.sync()
        G_optim.step(G_train_loss)
        if (batch_idx%100==0):
            print("train batch_idx",batch_idx,"epoch",epoch)
            print('  D_id training loss =', D_id_train_loss.data.mean())
            print('  D_se training loss =', D_se_train_loss.data.mean())
            print('  G training loss =', G_train_loss.data.mean())
    pass

train_epoch = 50

train(train_epoch)
