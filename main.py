import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from omegaconf import OmegaConf

from models.autoencoder import AutoEncoder, Encoder, Decoder
from data.augmenter import DataAugmenter
from train import Trainer
from utils.printer import Printer


if __name__ == "__main__":
    parser = ArgumentParser(description="train a model on MIDI data")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "--data_dir", default=None, help="directory of custom training data, in .npz"
    )
    parser.add_argument(
        "--param_file", default=None, help="path to parameter file, in .yaml"
    )
    parser.add_argument("--model", help="which model to train")
    args = parser.parse_args()
    params = OmegaConf.load(args.param_file)

    print(f"running with arguments:\n{args}")
    print(f"running with parameters:\n{params}")

    augmenter = DataAugmenter(args.data_dir, params.augmenter)
    dataset = augmenter.augment(True)
    loader = DataLoader(
        dataset,
        batch_size=params.loader.batch_size,
        shuffle=params.loader.shuffle,
        num_workers=params.loader.num_workers,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(params.encoder)
    decoder = Decoder(params.decoder)
    model = AutoEncoder().to(device)
    # model = Autoencoder(encoder, decoder).to(device)
    trainer = Trainer(model, params.model_name, loader, device, params.trainer)

    trainer.train()
    test_label, test_image = augmenter.get_clean()
    trainer.test_reconstruction(test_image, test_label, args.param_file)
