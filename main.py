import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf

from models.autoencoder import Autoencoder, Encoder, Decoder
from data.augmenter import DataAugmenter
from data.loader import CustomLoader
from train import Trainer


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
    print("running with arguments", args)

    params = OmegaConf.load(args.param_file)
    print("running with parameters", params)

    augmenter = DataAugmenter(args.data_dir, params.augmenter)
    dataset = augmenter.augment(True)

    loader = CustomLoader(dataset, params.loader)

    encoder = Encoder(params.encoder)
    decoder = Decoder(params.decoder)
    model = Autoencoder(encoder, decoder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = Trainer(model, params.model_name, loader, device, params.trainer)
