#!/usr/bin/env python

import argparse
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image into an ASCII art image with black shapes on colored cells."
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to save the output image",
        default="output_image.png",
    )
    parser.add_argument(
        "-n",
        "--num_clusters",
        type=int,
        default=10,
        help="Number of color clusters (shapes)",
    )
    parser.add_argument(
        "-s", "--cell_size", type=int, default=20, help="Size of each cell in pixels"
    )
    parser.add_argument(
        "-g",
        "--grid_line_width",
        type=int,
        default=1,
        help="Width of grid lines in pixels",
    )
    parser.add_argument(
        "--max_width", type=int, default=80, help="Maximum width of the image in pixels"
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=80,
        help="Maximum height of the image in pixels",
    )
    args = parser.parse_args()

    image_path = args.image_path
    output_path = args.output_path
    N = args.num_clusters
    cell_size = args.cell_size
    grid_line_width = args.grid_line_width
    max_size = (args.max_width, args.max_height)

    # Load image
    image = Image.open(image_path)
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image
    image.thumbnail(max_size, Image.LANCZOS)
    # Get pixel data
    pixel_data = np.array(image)
    h, w, _ = pixel_data.shape
    pixels = pixel_data.reshape(-1, 3)
    # Cluster colors
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    predicted_clusters = kmeans.predict(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    new_pixels = kmeans.cluster_centers_[predicted_clusters]

    shapes = [
        "■",
        "▲",
        "●",
        "◆",
        "♥",
        "♣",
        "♠",
        "☼",
        "☯",
        "☮",
        "☁",
        "✈",
        "✉",
        "☏",
        "☣",
        "☢",
        "♻",
        "✝",
        "✡",
        "☪",
        "☭",
        "★",
        "☆",
        "☹",
        "☺",
        "☻",
        "✌",
        "✋",
        "✊",
        "✋",
        "✌",
    ]
    if len(shapes) < N:
        print(f"Not enough shapes ({len(shapes)}) for {N} clusters.")
        sys.exit(1)
    shapes = shapes[:N]
    cluster_shapes = {i: shapes[i] for i in range(N)}
    # Reshape labels and pixels to image shape
    labels_2d = labels.reshape(h, w)
    new_pixels_2d = new_pixels.reshape(h, w, 3)
    # Create output image
    output_width = w * cell_size + (w + 1) * grid_line_width
    output_height = h * cell_size + (h + 1) * grid_line_width
    output_image = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(output_image)
    # Load font
    font_size = cell_size - 4  # Adjust font size to fit within the cell
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except IOError:
        # If the font is not found, use the default font
        font = ImageFont.load_default()
    # Draw cells and shapes
    for i in range(h):
        for j in range(w):
            # Calculate position
            x = j * (cell_size + grid_line_width) + grid_line_width
            y = i * (cell_size + grid_line_width) + grid_line_width
            # Get cluster label
            label = labels_2d[i, j]
            # Get shape
            shape = cluster_shapes[label]
            # Get pixel color
            color = tuple(new_pixels_2d[i, j].astype(int))
            # Fill the cell with the pixel color
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color)
            # Draw shape in black
            text_len = draw.textlength(shape, font=font)
            text_x = x + (cell_size - text_len) / 2
            text_y = y + (cell_size - text_len) / 2
            draw.text((text_x, text_y), shape, fill="black", font=font)
    # Draw grid lines
    for i in range(h + 1):
        y = i * (cell_size + grid_line_width)
        draw.line([(0, y), (output_width, y)], fill="black", width=grid_line_width)
    for j in range(w + 1):
        x = j * (cell_size + grid_line_width)
        draw.line([(x, 0), (x, output_height)], fill="black", width=grid_line_width)
    # Save output image
    output_image.save(output_path)
    print(f"Output image saved to {output_path}")

    root = tk.Tk()
    app = ImageEditorApp(
        root,
        output_path,
        args.cell_size,
        args.grid_line_width,
        cluster_shapes,
        cluster_centers,
        font,
    )
    root.mainloop()


class ImageEditorApp:
    def __init__(
        self,
        master,
        image_path,
        cell_size,
        grid_line_width,
        cluster_shapes,
        cluster_centers,
        font,
    ):
        self.master = master
        self.master.title("ASCII Image Editor")

        self.cell_size = cell_size
        self.grid_line_width = grid_line_width

        # Load the image
        self.image = Image.open(image_path)
        self.draw = ImageDraw.Draw(self.image)
        self.image_data = np.array(self.image)
        self.h, self.w, _ = self.image_data.shape

        # Get unique squares (colors)
        self.unique_colors = cluster_centers
        self.cluster_shapes = cluster_shapes
        self.selected_color = None
        self.font = font

        # Create the UI
        self.create_ui()

    def create_ui(self):
        # Create the legend
        self.legend_frame = tk.Frame(self.master)
        self.legend_frame.pack(side=tk.LEFT, fill=tk.Y)

        legend_label = tk.Label(self.legend_frame, text="Legend", font=("Arial", 14))
        legend_label.pack(pady=5)

        self.legend_canvas = tk.Canvas(self.legend_frame, width=100)
        self.legend_canvas.pack(side=tk.LEFT, fill=tk.Y)

        self.create_legend()

        # Create the image canvas
        self.canvas_frame = tk.Frame(self.master)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.draw_image()

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Add save button
        self.save_button = tk.Button(
            self.master, text="Save Image", command=self.save_image
        )
        self.save_button.pack(side=tk.BOTTOM, pady=10)

    def create_legend(self):
        self.legend_items = []
        y_position = 10
        for index, (color, shape) in enumerate(
            zip(self.unique_colors, self.cluster_shapes.values())
        ):
            rect = self.legend_canvas.create_rectangle(
                10,
                y_position,
                90,
                y_position + 30,
                fill=rgb_to_hex(color),
                outline="black",
            )
            self.legend_canvas.tag_bind(
                rect,
                "<Button-1>",
                lambda event, c=color, s=shape, index=index: self.on_legend_click(
                    c, s, index
                ),
            )
            y_position += 40
            self.legend_items.append(rect)

    def on_legend_click(self, color, shape, index):
        self.selected_color = color
        self.selected_shape = shape
        # Highlight the selected color
        for item in self.legend_items:
            self.legend_canvas.itemconfig(item, width=1)
        self.legend_canvas.itemconfig(self.legend_items[index], width=3)

    def draw_image(self):
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_canvas_click(self, event):
        if self.selected_color is None:
            return

        x = event.x
        y = event.y

        # Calculate the cell coordinates
        cell_x = int(x / (self.cell_size + self.grid_line_width))
        cell_y = int(y / (self.cell_size + self.grid_line_width))

        # Calculate the pixel coordinates in the image
        img_x = cell_x * (self.cell_size + self.grid_line_width) + self.grid_line_width
        img_y = cell_y * (self.cell_size + self.grid_line_width) + self.grid_line_width

        self.draw.rectangle(
            [img_x, img_y, img_x + self.cell_size, img_y + self.cell_size],
            fill=rgb_to_hex(self.selected_color),
        )
        # Draw shape in black
        print(self.selected_shape)
        text_len = self.draw.textlength(self.selected_shape, font=self.font)
        text_x = img_x + (self.cell_size - text_len) / 2
        text_y = img_y + (self.cell_size - text_len) / 2
        self.draw.text(
            (text_x, text_y),
            self.selected_shape,
            fill="black",
            font=self.font,
        )

        self.draw_image()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg;*.jpeg"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            # Draw grid lines
            for i in range(self.h + 1):
                y = i * (self.cell_size + self.grid_line_width)
                self.draw.line(
                    [(0, y), (self.w, y)],
                    fill="black",
                    width=self.grid_line_width,
                )
            for j in range(self.w + 1):
                x = j * (self.cell_size + self.grid_line_width)
                self.draw.line(
                    [(x, 0), (x, self.h)], fill="black", width=self.grid_line_width
                )
            self.image.save(file_path)
            messagebox.showinfo("Image Saved", f"Image saved to {file_path}")


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


if __name__ == "__main__":
    main()
