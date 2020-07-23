from keras.applications import VGG16
from keras.models import Model
import keras
from keras.layers import Dense, Flatten
from keras.datasets import cifar100
from keras import Input
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from kerassurgeon.operations import delete_channels

from utils import data_generator

import os
import math
import numpy as np

from cifarvgg import cifar100vgg


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Cifar10VGG16:

    def __init__(self, b=0.5):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)
        self.model = self.__build_model()
        self.num_classes = 100
        self.b = b
        self.action_size = None
        self.state_size = None
        self.epochs = 2
        self.base_model_accuracy = None
        self.action_size = None
        self.state_size = None
        self._current_state = 0
        self.layer_name = None
        train_gen = ImageDataGenerator(featurewise_std_normalization=True, featurewise_center=True)
        test_gen = ImageDataGenerator(featurewise_std_normalization=True, featurewise_center=True)
        train_gen.fit(self.x_train)
        test_gen.fit(self.x_test)
        self.train = train_gen.flow(self.x_train, self.y_train)
        self.test = test_gen.flow(self.x_test, self.y_test)


    def __build_model(self):
        """Builds the VGG16 Model
        """
        optm = SGD(learning_rate = 1e-4)
        model_obj = cifar100vgg.cifar100vgg(train=False)
        model = model_obj.getModel()
        model.compile(loss='sparse_categorical_crossentropy',optimizer=optm, metrics=['accuracy'])
        return model

    def get(self, layer_name='conv2d_1'):
        self.layer_name = layer_name
        x = self.model.get_layer(layer_name).get_weights()[0]
        self.state_size, self.action_size = x.shape[:3], x.shape[-1]
        x = x[:, :, :, self._current_state]
        if self._current_state + 1 == self.action_size:
            return True, x
        self._current_state += 1
        return False, x

    def _accuracy_term(self, new_model):
        new_model.fit_generator(self.train, epochs=self.epochs)
        p_hat = new_model.evaluate_generator(self.test)[0]
        if not self.base_model_accuracy:
            print('Calculating the accuracy of the base line model')
            self.base_model_accuracy = self.model.evaluate_generator(test)[0]
        accuracy_term = (self.b - (self.base_model_accuracy - p_hat)) / self.b
        return accuracy_term

    def step(self, action):
        action = np.where(action[0] == 0)[0]
        new_model = delete_channels(self.model, layer=self.model.get_layer(self.layer_name), channels=action)
        new_model.compile(loss="categorical_crrossentropy", optimizer='sgd', metrics=['accuracy'])
        reward = self._accuracy_term(new_model) - math.log10(self.action_size / len(action))
        done, x = self.get()
        return action, reward, done, x
