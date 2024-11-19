from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetPreparation:
    @staticmethod
    def prepare_datasets(train_dir, test_dir, target_size=(150, 150), batch_size=32, class_mode='binary'):
        train = ImageDataGenerator(rescale=1/255)
        test = ImageDataGenerator(rescale=1/255)
        
        train_dataset = train.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
        )
        
        test_dataset = test.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
        )
        return train_dataset, test_dataset
