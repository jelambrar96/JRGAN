import os
import datetime

import numpy as np
from matplotlib import pyplot as plt

from keras.applications import VGG19
from keras.optimizers import Adam
from keras.models import Model

from keras.layers import Input

from ..data.dataloader import DataLoader
from .basicmodel import BasicModel
from ..discriminator.factory import FactoryDis
from ..generator.factory import FactoryGen


class SRModel00(BasicModel):

    def __init__(self, input_shape, upscale, learning_rate, path_dataset,
                 filters_gen=32, filters_dis=32):
        # call superclass constructor
        print('srmodel construcor init')
        BasicModel.__init__(self, input_shape=input_shape, scale=upscale)
        #
        self._learning_rate = learning_rate
        self._optimizer = Adam(self._learning_rate, 0.5)

        self._vgg = self._build_vgg()
        self._vgg.trainable = False
        self._vgg.compile(loss='mse',
                          optimizer=self._optimizer,
                          metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = path_dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self._output_shape[0],
                                               self._output_shape[1]))

        # Calculate output shape of D (PatchGAN)
        patch = int(self._output_shape[0] / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self._dis = FactoryDis.getdis(0, self._output_shape,
                                      filters_dis, model=True)
        self._dis.compile(loss='mse', optimizer=self._optimizer,
                          metrics=['accuracy'])

        self._gen = FactoryGen.getgen(0, input_shape, filters_gen, model=True)
        fake_input_image = Input(shape=input_shape)
        fake_output = self._gen(fake_input_image)

        fake_output_image = Input(shape=self._output_shape)
        fake_features = self._vgg(fake_output)

        self._dis.trainable = False
        validity = self._dis(fake_output_image)

        self._combined = Model([fake_input_image, fake_output_image],
                               [validity, fake_features])
        self._combined.compile(loss=['binary_crossentropy', 'mse'],
                               loss_weights=[1e-3, 1],
                               optimizer=self._optimizer)

        print('srmodel construcor finish')

    def _build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image
        features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self._output_shape)
        # Extract image features
        img_features = vgg(img)
        return Model(img, img_features)

    """
    """
    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            # ----------------------
            #  Train Discriminator
            # ----------------------
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            # print(imgs_hr.shape)
            # print(imgs_lr.shape)
            # From low res. image generate high res. version
            fake_hr = self._gen.predict(imgs_lr)
            #
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self._dis.train_on_batch(imgs_hr, valid)
            d_loss_fake = self._dis.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ------------------
            #  Train Generator
            # ------------------
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)
            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self._vgg.predict(imgs_hr)
            # Train the generators
            g_loss = self._combined.train_on_batch([imgs_lr, imgs_hr],
                                                   [valid, image_features])
            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("%d time: %s" % (epoch, elapsed_time))
            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2,
                                                      is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)
        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()
        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name,
                                                       epoch, i))
            plt.close()
