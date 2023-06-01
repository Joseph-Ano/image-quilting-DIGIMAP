import typing
import abc

import numpy as np
import imageio


class BoxIndeces(typing.NamedTuple):  # See 'ImageQuiltingTemplate.extract_patch' for usage
    top: int
    bottom: int
    left: int
    right: int


class QuiltingOutputs(typing.NamedTuple):
    canvas_indeces: BoxIndeces
    canvas_patch: np.ndarray
    texture_indeces: BoxIndeces
    texture_patch: np.ndarray
    texture_mask: np.ndarray
    combined_patch: np.ndarray
    canvas: np.ndarray
    canvas_prev: np.ndarray


class ImageQuilting(abc.ABC):  # ABC means ABstract Class

    def __init__(self, texture_file: str):
        self.texture_image = self.load_image(texture_file)

    def load_image(self, path: str) -> np.ndarray:
        return np.array(imageio.imread(path)) / 255

    def save_image(self, path: str, image: np.ndarray):
        image = image * 255
        image = image.astype(np.uint8)
        return imageio.imsave(path, image)

    @abc.abstractmethod
    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces: ...

    @abc.abstractmethod
    def compute_texture_mask(
        self,
        canvas_patch: np.ndarray,
        texture_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        texture_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> np.ndarray: ...

    def generate_blank_canvas(self, output_height: int, output_width: int) -> np.ndarray:
        num_channels = self.texture_image.shape[-1]
        return np.zeros((output_height, output_width, num_channels), dtype=self.texture_image.dtype)

    def extract_patch(self, image: np.ndarray, indeces: BoxIndeces) -> np.ndarray:
        # Examine this code to understand the content of class BoxIndeces
        return image[indeces.top:indeces.bottom, indeces.left:indeces.right, :].copy()

    def iterate_block_indeces(self, height: int, width: int, block_size: int, block_overlap: int) -> typing.Iterable[BoxIndeces]:
        def generator():
            for top in range(0, height - block_overlap, block_size - block_overlap):
                bottom = min(top + block_size, height)
                for left in range(0, width - block_overlap, block_size - block_overlap):
                    right = min(left + block_size, width)
                    yield BoxIndeces(top, bottom, left, right)
        return list(generator())

    def iterate_texture_generation(self, block_size: int = None, block_overlap: int = None, canvas_size: typing.Union[int, typing.Tuple[int, int]] = None):
        texture_height, texture_width, *_ = self.texture_image.shape

        if block_size is None:
            block_size = int(min(texture_height, texture_width) / 20)  # Default is 1/20 size of source_texture
        if block_overlap is None:
            block_overlap = int(block_size / 6)  # Default is 1/6 size of block_size

        if canvas_size is None:
            canvas_size = 2 * texture_height, 2 * texture_width  # Default is ×2 size of source_texture
        if isinstance(canvas_size, int):
            canvas_size = canvas_size, canvas_size  # If single value convert to tuple, e.g. 2 → (2, 2)
        canvas_height, canvas_width = canvas_size

        canvas_image = self.generate_blank_canvas(output_height=canvas_height, output_width=canvas_width)

        for canvas_indeces in self.iterate_block_indeces(height=canvas_height, width=canvas_width, block_size=block_size, block_overlap=block_overlap):
            canvas_patch = self.extract_patch(canvas_image, canvas_indeces)

            texture_indeces = self.find_matching_texture_patch_indeces(
                canvas_patch=canvas_patch,
                canvas_indeces=canvas_indeces,
                canvas=canvas_image,
                block_size=block_size,
                block_overlap=block_overlap
            )
            texture_patch = self.extract_patch(self.texture_image, texture_indeces)

            texture_mask = self.compute_texture_mask(
                canvas_patch=canvas_patch,
                texture_patch=texture_patch,
                canvas_indeces=canvas_indeces,
                texture_indeces=texture_indeces,
                canvas=canvas_image,
                block_size=block_size,
                block_overlap=block_overlap
            )
            combined_patch = texture_patch * texture_mask[:, :, np.newaxis] + canvas_patch * (1 - texture_mask[:, :, np.newaxis])

            canvas_image_prev = canvas_image.copy()
            top, bottom, left, right = canvas_indeces
            canvas_image[top:bottom, left:right, :] = combined_patch

            yield QuiltingOutputs(
                canvas_indeces=canvas_indeces,
                canvas_patch=canvas_patch,
                texture_indeces=texture_indeces,
                texture_patch=texture_patch,
                texture_mask=texture_mask,
                combined_patch=combined_patch,
                canvas=canvas_image.copy(),
                canvas_prev=canvas_image_prev.copy()
            )

    def generate_texture(self, canvas_file=None, block_size: int = None, block_overlap: int = None, canvas_size: typing.Union[int, typing.Tuple[int, int]] = None) -> np.ndarray:
        process = self.iterate_texture_generation(
            block_size=block_size,
            block_overlap=block_overlap,
            canvas_size=canvas_size
        )

        canvas = None
        for outputs in process:
            canvas = outputs.canvas

        if canvas_file is not None:
            self.save_image(canvas_file, canvas)

        return canvas
