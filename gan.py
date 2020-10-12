from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread



class GanNetwork:
    def __init__(self):
        print("Initialization...")

        # Load in the data
        self.mnist = tf.keras.datasets.mnist
        
        # Dimensionality of the latent space
        self.latent_dim = 100

        # Config
        self.batch_size = 32
        self.epochs = 300



    def dataprep(self):
        print("Preparing the data")
        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()

        # map inputs to (-1, +1) for better training
        x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 - 1
        print("x_train.shape:", x_train.shape)

        # Flatten the data
        N, self.H, self.W = x_train.shape
        self.D = self.H * self.W
        x_train = x_train.reshape(-1, self.D)
        x_test = x_test.reshape(-1, self.D) 

        return x_train, y_train, x_test, y_test

    def build_generator(self, latent_dim):
        print("Building the generator")

        i = Input(shape=(latent_dim,))
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(self.D, activation='tanh')(x)

        model = Model(i, x)
        return model

    def build_discriminator(self, img_size):
        print("Building the discriminator")
        
        i = Input(shape=(img_size,))
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(i, x)
        return model

    def compiler(self):
        print("Compiling the models")

        # Compile both models in preparation for training

        # Build and compile the self.discriminator
        self.discriminator = self.build_discriminator(self.D)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

        # Build and compile the combined model
        self.generator = self.build_generator(self.latent_dim)

        # Create an input to represent noise sample from latent space
        z = Input(shape=(self.latent_dim,))

        # Pass noise through generator to get an image
        img = self.generator(z)

        # Make sure only the generator is trained
        self.discriminator.trainable = False

        # The true output is fake, but we label them real!
        fake_pred = self.discriminator(img)

        # Create the combined model object
        self.combined_model = Model(z, fake_pred)

        # Compile the combined model
        self.combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


    def train(self, x_train, y_train, x_test, y_test):
        print("Training the models!")
        # Train the GAN
        sample_period = 200 # every `sample_period` steps generate and save some data

        # Create batch labels to use when calling train_on_batch
        ones = np.ones(self.batch_size)
        zeros = np.zeros(self.batch_size)

        # Store the losses
        d_losses = []
        g_losses = []

        # Create a folder to store generated images
        if not os.path.exists('gan_images'):
            os.makedirs('gan_images')

        
        # Main training loop
        for epoch in range(self.epochs):
        
            ###########################
            ### Train self.discriminator ###
            ###########################

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_imgs = x_train[idx]

            # Generate fake images
            noise = np.random.randn(self.batch_size, self.latent_dim)
            fake_imgs = self.generator.predict(noise)

            # Train the self.discriminator
            # both loss and accuracy are returned
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_imgs, ones)
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(fake_imgs, zeros)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc  = 0.5 * (d_acc_real + d_acc_fake)


            #######################
            ### Train self.generator ###
            #######################

            noise = np.random.randn(self.batch_size, self.latent_dim)
            g_loss = self.combined_model.train_on_batch(noise, ones)

            # do it again!
            noise = np.random.randn(self.batch_size, self.latent_dim)
            g_loss = self.combined_model.train_on_batch(noise, ones)

            # Save the losses
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if epoch % 100 == 0:
                print(f"epoch: {epoch+1}/{self.epochs}, d_loss: {d_loss:.2f}, \
                      d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")

            if epoch % sample_period == 0:
                self.sample_images(epoch)
        

        #plots
        plt.plot(g_losses, label='g_losses')
        plt.plot(d_losses, label='d_losses')
        plt.legend()

        a = imread('gan_images/0.png')
        plt.imshow(a)
        
        #a = imread('gan_images/{}.png'.format(self.epochs//sample_period))
        #plt.imshow(a)


    def sample_images(self, epoch):
        rows, cols = 5, 5
        noise = np.random.randn(rows * cols, self.latent_dim)
        imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                axs[i,j].imshow(imgs[idx].reshape(self.H, self.W), cmap='gray')
                axs[i,j].axis('off')
                idx += 1
        fig.savefig("gan_images/%d.png" % epoch)
        plt.close()


    def run(self):
        print("Starting to run")
        x_train, y_train, x_test, y_test = self.dataprep()
        self.compiler()
        self.train(x_train, y_train, x_test, y_test)


if __name__=="__main__":
    gan_gen = GanNetwork()
    gan_gen.run()

