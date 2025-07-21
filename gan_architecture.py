#!/usr/bin/env python3
"""
CycleGAN Architecture for MRI T1-T2 Style Transfer
Based on analysis of sample projects and best practices

This implementation includes:
- U-Net Generator with skip connections
- PatchGAN Discriminator
- CycleGAN training loop with multiple loss functions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class InstanceNormalization(layers.Layer):
    """Instance Normalization layer for CycleGAN"""
    
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """Downsampling block for U-Net encoder"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                           kernel_initializer=initializer, use_bias=False))
    
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())
    
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsampling block for U-Net decoder"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))
    
    if norm_type.lower() == 'batchnorm':
        result.add(layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())
    
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    
    result.add(layers.ReLU())
    return result

def Generator(norm_type='instancenorm', target=None):
    """
    U-Net Generator for CycleGAN
    
    Args:
        norm_type: 'batchnorm' or 'instancenorm'
        target: Optional target for naming
    
    Returns:
        Generator model
    """
    inputs = layers.Input(shape=[256, 256, 1])
    
    # Encoder (Downsampling)
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm_type),                   # (bs, 64, 64, 128)
        downsample(256, 4, norm_type),                   # (bs, 32, 32, 256)
        downsample(512, 4, norm_type),                   # (bs, 16, 16, 512)
        downsample(512, 4, norm_type),                   # (bs, 8, 8, 512)
        downsample(512, 4, norm_type),                   # (bs, 4, 4, 512)
        downsample(512, 4, norm_type),                   # (bs, 2, 2, 512)
        downsample(512, 4, norm_type),                   # (bs, 1, 1, 512)
    ]
    
    # Decoder (Upsampling)
    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, norm_type),                      # (bs, 16, 16, 1024)
        upsample(256, 4, norm_type),                      # (bs, 32, 32, 512)
        upsample(128, 4, norm_type),                      # (bs, 64, 64, 256)
        upsample(64, 4, norm_type),                       # (bs, 128, 128, 128)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same',
                                kernel_initializer=initializer,
                                activation='tanh')  # (bs, 256, 256, 1)
    
    x = inputs
    
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x)
    
    model_name = f"generator_G" if target is None else f"generator_{target}"
    return keras.Model(inputs=inputs, outputs=x, name=model_name)

def Discriminator(norm_type='instancenorm', target=None):
    """
    PatchGAN Discriminator for CycleGAN
    
    Args:
        norm_type: 'batchnorm' or 'instancenorm'
        target: Optional target for naming
    
    Returns:
        Discriminator model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = layers.Input(shape=[256, 256, 1], name='input_image')
    
    x = inp
    
    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)    # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)    # (bs, 32, 32, 256)
    
    zero_pad1 = layers.ZeroPadding2D()(down3)       # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                        kernel_initializer=initializer,
                        use_bias=False)(zero_pad1)   # (bs, 31, 31, 512)
    
    if norm_type.lower() == 'batchnorm':
        norm1 = layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)
    
    leaky_relu = layers.LeakyReLU()(norm1)
    
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    
    last = layers.Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    
    model_name = f"discriminator_D" if target is None else f"discriminator_{target}"
    return keras.Model(inputs=inp, outputs=last, name=model_name)

class CycleGAN(keras.Model):
    """
    CycleGAN model for MRI T1-T2 style transfer
    """
    
    def __init__(
        self,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGAN, self).__init__()
        self.gen_g = generator_g
        self.gen_f = generator_f
        self.disc_x = discriminator_x
        self.disc_y = discriminator_y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
    def compile(
        self,
        gen_g_optimizer,
        gen_f_optimizer,
        disc_x_optimizer,
        disc_y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGAN, self).compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        
    def train_step(self, batch_data):
        # x is T1 images and y is T2 images
        real_x, real_y = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X
            
            fake_y = self.gen_g(real_x, training=True)
            fake_x = self.gen_f(real_y, training=True)
            
            # Cycle (T1 -> T2 -> T1) and (T2 -> T1 -> T2)
            cycled_x = self.gen_f(fake_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)
            
            # Identity mapping
            same_x = self.gen_f(real_x, training=True)
            same_y = self.gen_g(real_y, training=True)
            
            # Discriminator outputs
            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)
            
            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)
            
            # Generator losses
            gen_g_loss = self.generator_loss_fn(disc_fake_y)
            gen_f_loss = self.generator_loss_fn(disc_fake_x)
            
            # Cycle consistency losses
            cycle_loss_g = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_f = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle
            
            # Identity losses
            id_loss_g = self.identity_loss_fn(real_y, same_y) * self.lambda_cycle * self.lambda_identity
            id_loss_f = self.identity_loss_fn(real_x, same_x) * self.lambda_cycle * self.lambda_identity
            
            # Total generator losses
            total_loss_g = gen_g_loss + cycle_loss_g + id_loss_g
            total_loss_f = gen_f_loss + cycle_loss_f + id_loss_f
            
            # Discriminator losses
            disc_x_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)
        
        # Calculate gradients
        grads_g = tape.gradient(total_loss_g, self.gen_g.trainable_variables)
        grads_f = tape.gradient(total_loss_f, self.gen_f.trainable_variables)
        grads_disc_x = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        grads_disc_y = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)
        
        # Apply gradients
        self.gen_g_optimizer.apply_gradients(zip(grads_g, self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(grads_f, self.gen_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(grads_disc_x, self.disc_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(grads_disc_y, self.disc_y.trainable_variables))
        
        return {
            "G_loss": total_loss_g,
            "F_loss": total_loss_f,
            "D_X_loss": disc_x_loss,
            "D_Y_loss": disc_y_loss,
        }

def generator_loss(fake):
    """Generator loss function"""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake, labels=tf.ones_like(fake)))

def discriminator_loss(real, fake):
    """Discriminator loss function"""
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real, labels=tf.ones_like(real)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake, labels=tf.zeros_like(fake)))
    return (real_loss + fake_loss) * 0.5

def create_cyclegan_model():
    """
    Create and compile CycleGAN model
    
    Returns:
        Compiled CycleGAN model
    """
    # Create generators
    gen_g = Generator(norm_type='instancenorm', target='T1_to_T2')  # T1 -> T2
    gen_f = Generator(norm_type='instancenorm', target='T2_to_T1')  # T2 -> T1
    
    # Create discriminators
    disc_x = Discriminator(norm_type='instancenorm', target='T1')   # T1 discriminator
    disc_y = Discriminator(norm_type='instancenorm', target='T2')   # T2 discriminator
    
    # Create CycleGAN model
    cycle_gan_model = CycleGAN(
        generator_g=gen_g,
        generator_f=gen_f,
        discriminator_x=disc_x,
        discriminator_y=disc_y
    )
    
    # Compile model
    cycle_gan_model.compile(
        gen_g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_f_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_x_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss,
        disc_loss_fn=discriminator_loss,
    )
    
    return cycle_gan_model

if __name__ == "__main__":
    # Test model creation
    print("Creating CycleGAN model...")
    model = create_cyclegan_model()
    print("Model created successfully!")
    
    # Print model summaries
    print("\n=== Generator G (T1 -> T2) Summary ===")
    model.gen_g.summary()
    
    print("\n=== Generator F (T2 -> T1) Summary ===")
    model.gen_f.summary()
    
    print("\n=== Discriminator X (T1) Summary ===")
    model.disc_x.summary()
    
    print("\n=== Discriminator Y (T2) Summary ===")
    model.disc_y.summary()

