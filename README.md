# imageToStitch
An Image to cross stitch pattern interface which uses clustering to create a cross-stitch pattern from an image.
It then gives you an interface to edit that pattern to make any fixes you need.

# Requirements

This is a python script that uses tkinter. Depending on your system you may need additional packages, and you will also need to install the requirements with:

```
python -m pip install -r requirements.txt
```

# Usage

Example usage with the cat image from: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJIvxvHU40AcXkLlwUJ7jdj3YBYV3O25wmNQ&s - which I saved as cat.png.

The un-edited output is checked in to the repo as `output_image.png`.

```
python image_to_stitch.py /path/to/cat.png
```

This will output the image size, and the generated cross-stitch pattern.
It will then open a tkinter window for you to edit the result.


There are additional arguments for:
 - the number of clusters (colors) to use
 - output path
 - resizing the image before generating the pattern
 - the cell-size used when displaying the image in the editing UI


Full help output from `python image_to_stitch.py -h`

```
usage: image_to_stitch.py [-h] [-o OUTPUT_PATH] [-n NUM_CLUSTERS] [-s CELL_SIZE] [-g GRID_LINE_WIDTH] [--width WIDTH]
                          [--height HEIGHT]
                          image_path

Convert an image into an ASCII art image with black shapes on colored cells.

positional arguments:
  image_path            Path to the input image

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save the output image
  -n NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                        Number of color clusters
  -s CELL_SIZE, --cell_size CELL_SIZE
                        Size of each cell in pixels, when editing the pattern.
  -g GRID_LINE_WIDTH, --grid_line_width GRID_LINE_WIDTH
                        Width of grid lines in pixels, when editing the pattern
  --width WIDTH         Resize the image to the specified width before generating the pattern
  --height HEIGHT       Resize the image to the specified height before generating the pattern
```
