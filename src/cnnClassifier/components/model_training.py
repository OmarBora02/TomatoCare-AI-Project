import os 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
import time
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )

        if self.config.params_is_augmented:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator=valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs
        )


    @staticmethod
    def save_model(path: Path, model:tf.keras.Model):
        model.save(path)


    
    ## Training function
    def train(self):
        if not tf.executing_eagerly():
            tf.config.run_functions_eagerly(True)

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )


        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps
            
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

