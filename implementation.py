from base import ImageQuilting, BoxIndeces, QuiltingOutputs

import typing
import numpy as np
import cv2
import scipy.ndimage as ndimage
import PIL.Image as Image
import imageio
import os

# SUBMISSION: You only need to submit this file 'implementation.py'

# Hint: You could refer to online sources for help as long as you cite it
# Hint: The entire implementation logic can be viewed in 'base.py'
# hint: Feel free to add as much helper functions in the class

# SCORE -1 if you imported additional modules
# SCORE +5 for submitting this file (bonus)


class ImageQuilting_AlgorithmAssignment(ImageQuilting):

    def get_overlap(self, patch: np.ndarray, canvas_indeces: BoxIndeces, block_size: int, block_overlap: int) -> np.ndarray:
        _, _, color = patch.shape

        row = canvas_indeces.top
        col = canvas_indeces.left

        if(row == 0):
            overlap_section = patch[:, :block_overlap, :]
        elif(col == 0):
            overlap_section = patch[:block_overlap, :, :]
        else:
            overlap_section = np.zeros((block_size, block_size, color), dtype=float)
            overlap_section[:, :block_overlap, :] = patch[:, :block_overlap, :]
            overlap_section[:block_overlap, :, :] = patch[:block_overlap, :, :]
            
        return overlap_section

    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces:

        # SCORE +1 by iterating through all other possible patch locations
        def enumerate_other_indeces() -> typing.Sequence[BoxIndeces]:
            # Hint: use self.texture_image as reference
            texture_height, texture_width, _ = self.texture_image.shape

            for top in range(0, texture_height - block_size):
                bottom = top + block_size
                for left in range(0, texture_width - block_size):
                    right = left + block_size

                    yield BoxIndeces(
                        top,
                        bottom,
                        left,
                        right
                    )

        # SCORE +1 by finding the overlap area in canvas_patch
        def extract_canvas_overlap() -> np.ndarray:
            # Hint: use canvas_patch, canvas_indeces, block_overlap
            overlap_section = self.get_overlap(canvas_patch, canvas_indeces, block_size, block_overlap)
        
            return overlap_section

        # SCORE +1 by finding the overlap area in patch
        def extract_overlap(patch: np.ndarray) -> np.ndarray:
            # Hint: use canvas_indeces, block_overlap
            overlap_section = self.get_overlap(patch, canvas_indeces, block_size, block_overlap)
                
            return overlap_section
        
        # SCORE +1 by finding the errors between between the overlapping patches
        def error_metric(a: np.ndarray, b: np.ndarray) -> float:
            # Hint: Use L2 loss ((a - b)**2).sum()

            # TODO: Replace this RANDOM implementation
            return ((a - b)**2).sum()

        canvas_overlap = extract_canvas_overlap()

        least_error: float = float("inf")
        least_error_indeces: BoxIndeces = None

        for other_indeces in enumerate_other_indeces():
            other_patch = self.extract_patch(self.texture_image, other_indeces)
            other_overlap = extract_overlap(other_patch)
        
            error = error_metric(canvas_overlap, other_overlap)
            
            if error < least_error:
                least_error = error
                least_error_indeces = other_indeces

        return least_error_indeces

    def compute_texture_mask(
        self,
        canvas_patch: np.ndarray,
        texture_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        texture_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> np.ndarray:
        texture_height, texture_width, _ = texture_patch.shape
        mask = np.ones((texture_height, texture_width))

        # SCORE +2 for implementing the core logic for finding the cut the minimizes the L2 error (minCut)
        # Hint: The implementation for minCut differs slightly for Left, Top, and Left-Top overlaps (see 'guide.jpg')
        # Hint: But as long as the core logic to at least one case is implemented, I'll give the points already

        # This code segment (minCut_left) was referenced from chat GPT
        def minCut_left(canvas_overlap: np.ndarray, texture_overlap: np.ndarray) -> np.ndarray:
            l2loss = np.sum((canvas_overlap-texture_overlap) ** 2, axis=2)
            height, width = l2loss.shape

            for row in range(1, height):
                for col in range(width):
                    left = max(0, col-1)
                    right = min(width-1, col+1)
                    prev_row = row - 1

                    l2loss[row, col] += min(l2loss[prev_row, left], l2loss[prev_row, col], l2loss[prev_row, right])

            path_index = []
            minimum = np.argmin(l2loss[-1])
            path_index.append(minimum)
           
            for row in range(height-1, 0, -1):
                left = max(0, minimum-1)
                right = min(width-1, minimum+1)
                prev_row = row - 1

                minimum = np.argmin(l2loss[prev_row, left:right+1]) + left
                path_index.append(minimum)
                
            return path_index[::-1]
        
        
        def minCut_top(canvas_overlap: np.ndarray, texture_overlap: np.ndarray) -> np.ndarray:
            l2loss = np.sum((canvas_overlap-texture_overlap) ** 2, axis=2)
            height, width = l2loss.shape

            for col in range(1, width):
                for row in range(height):
                    top = max(0, row-1)
                    bottom = min(row+1, height-1)
                    prev_col = col - 1

                    l2loss[row, col] += min(l2loss[top, prev_col], l2loss[row, prev_col], l2loss[bottom, prev_col])

            path_index = []
            minimum = np.argmin(l2loss[-1])
            path_index.append(minimum)
           
            for col in range(width-1, 0, -1):
                top = min(0, row-1)
                bottom = max(row+1, height-1)
                prev_col = col - 1

                minimum = np.argmin(l2loss[top:bottom+1, prev_col]) + top
                path_index.append(minimum)
                
            return path_index[::-1]

        canvas_overlap = self.get_overlap(canvas_patch, canvas_indeces, block_size, block_overlap)
        texture_overlap = self.get_overlap(texture_patch, canvas_indeces, block_size, block_overlap)
       
        # SCORE +1 for returning the correct mask for the initialization (row=0, column=0)
        if canvas_indeces.top == 0 and canvas_indeces.left == 0:
            return mask

        # SCORE +1 for returning the correct mask for Case 1: Left Overlap (row=0, column>1)
        if canvas_indeces.top == 0 and canvas_indeces.left != 0:
            path_index = minCut_left(canvas_overlap, texture_overlap)

            for y in range(texture_height):
                mask[y, :path_index[y]] = 0

            return mask

        # SCORE +1 for returning the correct mask for Case 2: Top Overlap (row>1, column=0)
        if canvas_indeces.top != 0 and canvas_indeces.left == 0:
            path_index = minCut_top(canvas_overlap, texture_overlap)

            for x in range(texture_width):
                mask[:path_index[x], x] = 0

            return mask

        # SCORE +1 for returning the correct mask for Case 3: Top-Left Overlap (row>1, column>1)
        if canvas_indeces.left != 0 and canvas_indeces.top != 0:
            vertical_path_index = minCut_left(canvas_overlap[:, :block_overlap, :], texture_overlap[:, :block_overlap, :])

            for y in range(texture_height):
                mask[y, :vertical_path_index[y]] = 0

            horizontal_path_index = minCut_top(canvas_overlap[:block_overlap, :, :], texture_overlap[:block_overlap, :, :])

            for x in range(texture_width):
                mask[:horizontal_path_index[x], x] = 0

            return mask

        raise Exception("This code should not run")
