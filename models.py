import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Reshape,
                                     BatchNormalization, LeakyReLU,
                                     Conv2DTranspose, Activation, 
                                     Conv2D, Flatten)

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import gradients

from loss import wasserstein_loss

class GAN:

    def __init__(self, latent_size=512):
        self.latent_size = latent_size
        self.img_size = (256, 256, 3)

        self.gen = self.__build_generator()
        self.dis = self.__build_discriminator()
        self.gan = self.__build_gan()

    def __build_generator(self):
        ins = Input((self.latent_size,))
        x = Dense(4 * 4 * self.latent_size,
                  kernel_initializer=RandomNormal(stddev=0.02))(ins)
        x = Reshape(target_shape(4, 4, self.latent_size))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 2
        x = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 3
        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 4
        x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Layer 5
        x = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Layer 6
        x = Conv2DTranspose(16, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Layer out
        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalizaiton()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Activation('tanh')(x)

        model = Model(inputs=ins, outputs=x)
        return model

    def __build_critic(self):
        ins = Input(self.img_size)
        
        x = Conv2D(16, kernel_size=5, strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(ins)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(32, kernel_size=5, strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(64, kernel_size=5, strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(128, kernel_size=5, strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(256, kernel_size=5, strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = Activation('softmax')(x)

        model = Model(inputs=ins, outputs=x)
        return model

    def __build_gan(self):

        opt = RMSprop(lr=0.00005)

        self.dis.compile(optimizer=opt, loss=wasserstein_loss)
        gan = Sequential([self.gen, self.dis])
        gan.compile(optimizer=opt, loss=wasserstein_loss)
        return gan

    def train(self,
              generator,
              epochs=15,
              alpha=0.00005,
              c=0.01,
              m=64,
              n_critic=5):
        """
        Train the WGAN

        Keyword arguments:
        epochs   -- number of training epochs (default 10)
        alpha    -- learning rate (default 0.00005)
        c        -- clipping parameter (default 0.01)
        n_critic -- number of iterations of the critic pre generator iteration
        """
       
        real_label = np.ones((m,1))
        fake_label = -1 * np.ones((m,1))

        for e in range(epochs):
            for _ in range(n_critic):
                # Sample points
                zs = self.gen.predict_on_batch(np.random.normal(size=(m, self.latent_size)))
                xs, _ = generator.next()

                # Calulate Gradient and update weights
                # TODO this is not the proper way to do it!!!
                #  second gradient doesnt make sense since it first 
                #  gets updated by train_on_batch with reals
                # Train with reals
                self.dis.train_on_batch(xs, real_label)

                # Train with fakes
                self.dis.train_on_batch(zs, fake_label)

                # Clip Weights - TODO improved wasserstein
                for layer in self.dis.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -c, c) for w in weights]
                    layer.set_weights(weights)

            self.dis.trainable = False
            # Train Generator
            zs = self.gen.predict_on_batch(np.random.normal(size=(m, self.latent_size)))
            self.gan.train_on_batch(zs, real_label)

            self.dis.trainable = True
