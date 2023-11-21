import os

from PIL import Image, ImageDraw, ImageFont


def crop_concat_add_text(image_paths, texts, output_path):
    cropped_images = []
    gap_width = 10
    text_position = "right bottom"

    for path, text in zip(image_paths, texts):
        img = Image.open(path)

        # Crop the image
        width, height = img.size
        cropped_img = img.crop((width / 4, 0, 3 * width / 4, height))

        # Add text to the cropped image
        draw = ImageDraw.Draw(cropped_img)
        font = ImageFont.truetype("arial.ttf", int(width * 0.05))
        text_width, text_height = draw.textsize(text, font=font)

        # Calculate text position for right bottom corner
        x_text = cropped_img.width - text_width - 10  # 10 pixels from the right edge
        y_text = cropped_img.height - text_height - 10  # 10 pixels from the bottom edge
        draw.text((x_text, y_text), text, fill=(255, 255, 255), font=font)  # White text

        cropped_images.append(cropped_img)

    # Create a white image for the gap
    height = max(img.height for img in cropped_images)
    gap = Image.new('RGB', (gap_width, height), (255, 255, 255))

    # Concatenate images with gaps
    total_width = sum(img.width for img in cropped_images) + gap_width * (len(cropped_images) - 1)
    combined_img = Image.new('RGB', (total_width, height))

    x_offset = 0
    for img in cropped_images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
        if x_offset < total_width:
            combined_img.paste(gap, (x_offset, 0))
            x_offset += gap_width

    combined_img.save(output_path)


# Example usage
image_dir = '/home/zwyan/3d_cv/repos/gaussian-splatting/output/bicycle/base/viewer'
image_names = ['23_1', '23_2', '23_3']
image_paths = [os.path.join(image_dir, image_name + '.png') for image_name in image_names]
texts = ["2x", "4x", "8x"]
output_name = 'concat.png'
output_path = os.path.join(image_dir, output_name)
crop_concat_add_text(image_paths, texts, output_path)
