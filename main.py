from base import ImageQuilting, BoxIndeces
from implementation import ImageQuilting_AlgorithmAssignment

import argparse
import os


def visualize_process(quiltor: ImageQuilting, output_dir: str, **kwds):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def draw_box(ax: plt.Axes, indeces: BoxIndeces):
        xy = (indeces.left, indeces.top)
        height = indeces.bottom - indeces.top
        width = indeces.right - indeces.left
        rect = patches.Rectangle(xy, height, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if not os.path.exists(output_dir):
        print(f"Creating a output folder at '{os.path.abspath(output_dir)}'")
        os.makedirs(output_dir)

    canvas = None

    for step, outputs in enumerate(quiltor.iterate_texture_generation(**kwds)):
        canvas = outputs.canvas

        fig, axs = plt.subplots(2, 3)

        ax: plt.Axes = axs[0, 0]
        ax.set_title("Canvas")
        ax.imshow(outputs.canvas_prev)
        draw_box(ax, outputs.canvas_indeces)

        ax: plt.Axes = axs[1, 0]
        ax.set_title("Source texture")
        ax.imshow(quiltor.texture_image)
        draw_box(ax, outputs.texture_indeces)

        ax: plt.Axes = axs[0, 1]
        ax.imshow(outputs.canvas_patch)
        ax.set_title("Query patch")

        ax: plt.Axes = axs[1, 1]
        ax.imshow(outputs.texture_patch)
        ax.set_title("Best match patch")

        ax: plt.Axes = axs[0, 2]
        ax.imshow(outputs.texture_mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Texture Mask")

        ax: plt.Axes = axs[1, 2]
        ax.imshow(outputs.combined_patch)
        ax.set_title("Combined patch")

        fig.suptitle(f"Currently Generating: {outputs.canvas_indeces}")

        filepath = os.path.join(output_dir, f"{step:09}.png")
        plt.savefig(filepath)
        plt.close()

    return canvas

# Run 'python main.py'
# to generate a new texture from 'sample_pattern.jpg'

# Run 'python main.py --visualize'
# to generate a new texture from 'sample_pattern.jpg' and visualize the process

# Run 'python main.py --image Path/To/Source/Texture --output Path/To/Generated/Texture'
# to generate a new texture from a source texture

# Run 'python main.py --visualize --image Path/To/Source/Texture --output Path/To/Generated/Texture'
# to generate a new texture from a source texture and visualize the process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Runs the image quilting algorithm that uses randomly generates a texture from a static image")
    parser.add_argument("--image", type=str, default="sample_pattern.jpg", help="File path of the source texture")
    parser.add_argument("--output_dir", type=str, default="_temp-visualization", help="Directory where the processing visualizations will be stored")
    parser.add_argument("--output", type=str, default="_temp-output.jpg", help="File path where the generated texture will be saved")
    parser.add_argument("--block_size", type=int, default=40, help="See 'guide.jpg'")
    parser.add_argument("--block_overlap", type=int, default=10, help="See 'guide.jpg'")
    parser.add_argument("--output_size", type=int, default=160, help="Size of output image")

    args = parser.parse_args()

    quiltor = ImageQuilting_AlgorithmAssignment(args.image)

    if args.visualize:
        print(f"Visualization will be stored in the directory '{args.output_dir}'")
        generated_texture = visualize_process(
            quiltor,
            output_dir=args.output_dir,
            block_size=args.block_size,
            block_overlap=args.block_overlap,
            canvas_size=args.output_size
        )
    else:
        generated_texture = quiltor.generate_texture(
            block_size=args.block_size,
            block_overlap=args.block_overlap,
            canvas_size=args.output_size
        )

    print(f"Saving image in'{args.output}'")
    quiltor.save_image(args.output, generated_texture)
