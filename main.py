import torch
import os
import json
import datetime
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloader
from data.transformation import train_transform, val_transform
from models.MIML import MLP, CombinedModel
from utils import device
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging

def main():
    project = Project()
    params = {
        'lr': 5e-5,
        'batch_size': 64,
        'epochs': 1000,
        'model': 'T4vsT8',
        'train_resnet': True  # Allows controlling trainability of ResNet from params
    }

    # Log device usage
    logging.info(f'Using device={device} 🚀')

    class_mapping = {
        'T4': 0,
        'T8': 1
        }

    # Data loading
    train_dl, val_dl, test_dl = get_dataloader( train_dir=os.path.join(project.data_dir, "TL-T4vsT8/train_mod"),
                                                val_dir=os.path.join(project.data_dir, "TL-T4vsT8/test_mod"),
                                                train_csv=os.path.join(project.data_dir, "TL-T4vsT8/normalized_feature_summary_train.csv"),
                                                val_csv=os.path.join(project.data_dir, "TL-T4vsT8/normalized_feature_summary_val.csv"),
                                                class_mapping = class_mapping,
                                                train_transform=train_transform,
                                                val_transform=val_transform,
                                                split=(0.6, 0.4),
                                                batch_size=params['batch_size'],
                                                pin_memory=True,
                                                num_workers=8)
    

    # Initialize Comet Experiment
    with open('secrets.json') as f:
        secrets = json.load(f)
    experiment = Experiment(api_key=secrets['COMET_API_KEY'], project_name="multiplex-iml", workspace="khayrulbuet13")
    experiment.log_parameters(params)
    experiment.set_name(params['model'])

    # Model setup
    mlp = MLP(input_size=3, hidden_size=32, output_size=16)
    model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=params['train_resnet']).to(device)


    # Load existing model if available
    model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    if os.path.exists(model_saved_path):
        model.load_state_dict(torch.load(model_saved_path))
        logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    logging.info(summary(model.resnet18, input_size=(3, 64, 64)))
    logging.info(summary(mlp, input_size=(3,)))

    # Optimizer and training configuration
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    poutyne_model = Model(model, optimizer, "cross_entropy", batch_metrics=["accuracy"]).to(device)

    # Callbacks
    current_time = datetime.datetime.now().strftime('%d %B %H:%M')
    checkpoint_path = os.path.join(project.checkpoint_dir, f"{current_time}-{params['model']}.pt")
    callbacks = [
        ReduceLROnPlateau(monitor="val_acc", patience=20, verbose=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=True),
        EarlyStopping(monitor="val_acc", patience=40, mode='max'),
        CometCallback(experiment)
    ]

    # Training
    poutyne_model.fit_generator(train_dl, val_dl, epochs=params['epochs'], callbacks=callbacks)

    # Evaluation
    loss, test_acc = poutyne_model.evaluate_generator(test_dl)
    logging.info(f'Test Accuracy={test_acc}')
    experiment.log_metric('test_acc', test_acc)

if __name__ == '__main__':
    main()
