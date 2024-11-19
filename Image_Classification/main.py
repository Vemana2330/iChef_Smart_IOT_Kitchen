from dataset_preparation import DatasetPreparation
from model import InstaChefModel
from trainer import Trainer
from predictor import Predictor

# File paths
TRAIN_DIR = "C:\Users\Image_Classification\Dataset\Train"
TEST_DIR = "C:\Users\Image_Classification\Dataset\Test"
TARGET_SIZE = (150, 150)
BATCH_SIZE = 32

# Step 1: Prepare datasets
train_dataset, test_dataset = DatasetPreparation.prepare_datasets(TRAIN_DIR, TEST_DIR, TARGET_SIZE, BATCH_SIZE)

# Step 2: Build model
model = InstaChefModel.build_model(input_shape=(150, 150, 3))

# Step 3: Train model
history = Trainer.train_model(model, train_dataset, test_dataset, epochs=8, steps_per_epoch=15)

# Step 4: Evaluate model
Trainer.evaluate_model(model, test_dataset)

# Step 5: Plot metrics
Trainer.plot_metrics(history)

# Step 6: Predict images
image_paths = [
    "/content/drive/MyDrive/InstaChef/Dataset/Burnt442/burnt_308.jpg",
    "/content/drive/MyDrive/InstaChef/Dataset/Cooked442/cooked_15.jpg"
]

for img_path in image_paths:
    result = Predictor.predict_image(model, img_path)
    print(f"{img_path.split('/')[-1]}: {result}")
