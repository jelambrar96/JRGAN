
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from .basicmodel import BasicModel




class SRGAN(BasicModel):

    def __init__(self, *args, **kwargs):
        BasicModel.__init__(self, *args, **kwargs)
        self.n_residual_blocks = 10
        self._df = 
        self._gf = 

        self._optimizer = Adam(self._learning_rate, 0.5)
        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=self._optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self._output_shape[0], self._output_shape[1]))
        # Calculate output shape of D (PatchGAN)
        patch = int(self._output_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        # self.gf = 64
        # self.df = 64
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        # Build the generator
        self.generator = self.build_generator()
        # High res. and low res. images
        img_hr = Input(shape=self._input_shape)
        img_lr = Input(shape=self._output_shape)
        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)
        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=self._optimizer)


    def _build_discriminator(self):
        #
        def d_block(layer_input, filters, strides=1, bn=True):
            # """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        # Input img
        d0 = Input(shape=self.hr_shape)
        # 
        d1 = d_block(d0, self._df, bn=False)
        d2 = d_block(d1, self._df, strides=2)
        d3 = d_block(d2, self._df*2)
        d4 = d_block(d3, self._df*2, strides=2)
        d5 = d_block(d4, self._df*4)
        d6 = d_block(d5, self._df*4, strides=2)
        d7 = d_block(d6, self._df*8)
        d8 = d_block(d7, self._df*8, strides=2)
        # 
        d9 = Dense(self._df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        # 
        return Model(d0, validity)
        
    def _build_generador(self):
        # create an residual
        def residual_block(layer_input, filters):
            # """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d
        # create an deconv 
        def deconv2d(layer_input):
            # """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u
        #
        # Low resolution image input
        img_lr = Input(shape=self._input_shape)
        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
        # Propogate through residual blocks
        r = residual_block(c1, self._gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self._gf)
        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
        return Model(img_lr, gen_hr)

    def build_vgg(self):
        # """
        # Builds a pre-trained VGG19 model that outputs image features extracted at the
        # third block of the model
        # """
        vgg = VGG19(weights='imagenet')
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self.hr_shape)
        # Extract image features
        img_features = vgg(img)
        return Model(img, img_features)


    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)


    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
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
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()