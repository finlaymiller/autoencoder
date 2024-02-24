import torch
import torch.nn.functional as F

import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path


class DataAugmenter:
    def __init__(self, input_dir, output_dir, params) -> None:
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir

        print(f"initialized DataAugmenter with input dir {self.input_dir}", params)

    def augment(self) -> str:
        for file in os.listdir(self.input_dir):
            outfile_name = f"{Path(file).stem}_p{self.params.num_permutations}_n{str(self.params.noise_factor).split('.')[1]}{'_v' if self.params.vshift else ''}"

            if not os.path.exists(os.path.join(self.output_dir, outfile_name)):
                self.augment_data(os.path.join(self.input_dir, file), outfile_name)
            else:
                print(
                    f"Augmented data found matching the provided settings {outfile_name}"
                )

        return os.path.join(self.output_dir, outfile_name)

    def augment_data(self, input_file, output_file) -> None:
        """Augments a set of passed-in images by a factor of 2*num_permutations"""
        clean_images = np.load(input_file)
        shifted_images = []
        noisy_images = []

        for name, image in tqdm(
            list(clean_images.items()), unit="images", dynamic_ncols=True
        ):
            time_factor = image[:, 0]  # save time factor
            image = np.delete(image, 0, axis=1)  # remove it from the image though
            if self.params.vshift:
                # vertical shift images
                shifted_images.append(
                    self.shift_image_vertically(
                        name, image, self.params.num_permutations
                    )
                )
            else:
                # reformat clean image array
                shifted_images.append([(name, image)])

            # add noise to images
            for si in shifted_images[-1]:
                new_key, shifted_image = si
                for _ in range(self.params.num_permutations):
                    # normalize
                    noisy_image = shifted_image / np.max(shifted_image)

                    # corrupt
                    noisy_image = torch.from_numpy(
                        noisy_image
                    ) + self.params.noise_factor * torch.randn(noisy_image.shape)

                    # reformat
                    noisy_image = self.format_image(noisy_image)

                    noisy_images.append((new_key, noisy_image))

        random.shuffle(noisy_images)

        np.savez_compressed(
            os.path.join(self.output_dir, output_file),
            **{name: arr for name, arr in noisy_images},
        )

        print(
            f"used {len(list(clean_images.keys()))} clean images to generate {len(noisy_images)} noisy images of shape {noisy_images[0][1].size()}"
        )

    def shift_image_vertically(self, name, array, num_iterations):
        shifted_images = []

        def find_non_zero_bounds(arr):
            # Find the first and last row index with a non-zero element
            rows_with_non_zero = np.where(arr.any(axis=1))[0]
            if len(rows_with_non_zero) == 0:
                return (0, arr.shape[0] - 1)  # Handle case with no non-zero elements
            return rows_with_non_zero[0], rows_with_non_zero[-1]

        def shift_array(arr, up=0, down=0):
            # Shift array vertically
            if up > 0:
                arr = np.roll(arr, -up, axis=0)
                arr[-up:] = 0
            elif down > 0:
                arr = np.roll(arr, down, axis=0)
                arr[:down] = 0
            return arr

        highest, lowest = find_non_zero_bounds(array)
        maximum_up = highest
        maximum_down = array.shape[0] - lowest - 1

        for _ in range(num_iterations):
            # Shift up and then down, decreasing the shift amount in each iteration
            for i in range(maximum_up, 0, -1):
                new_key = f"{Path(name).stem}_u{i:02d}"
                shifted_images.append((new_key, np.copy(shift_array(array, up=i))))
            for i in range(maximum_down, 0, -1):
                new_key = f"{Path(name).stem}_d{i:02d}"
                shifted_images.append((new_key, np.copy(shift_array(array, down=i))))

        random.shuffle(shifted_images)

        return shifted_images[:num_iterations]

    def format_image(self, image, remove_time=False):
        if remove_time:
            image = np.delete(image, 0, axis=1)
        image = torch.from_numpy(np.expand_dims(image, 0)).to(torch.float32)
        if torch.any(image > 1.0):
            image = image / image.max()
        image = F.pad(input=image, pad=(0, 12, 1, 1), mode="constant", value=0.0)

        return image
