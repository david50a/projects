from PIL import Image

# Define ASCII characters to represent different intensity levels
ASCII_CHARS = "@%#*+=-:. "


class ASCIIArtConverter:
    def __init__(self, image_path, width=450):
        self.image = Image.open(image_path)
        # Resize image to match desired width while maintaining aspect ratio
        aspect_ratio = self.image.height / self.image.width
        self.height = int(width * aspect_ratio)
        self.image = self.image.resize((width, self.height))
        # Convert image to RGB mode to retain color information
        self.image = self.image.convert("RGB")

    def image_to_ascii(self):
        ascii_art = ""
        colors = []  # Array to store colors
        for y in range(self.height):
            for x in range(450):
                # Get RGB values of the pixel
                r, g, b = self.image.getpixel((x, y))
                # Append color to the array
                colors.append((r, g, b))
                # Convert RGB values to grayscale intensity
                grayscale = int(0.21 * r + 0.72 * g + 0.07 * b)
                # Calculate ASCII character index based on intensity
                char_index = int((grayscale / 255) * (len(ASCII_CHARS) - 1))
                # Append ASCII character representing intensity
                ascii_art += ASCII_CHARS[char_index]
            ascii_art += "\n"
        return ascii_art, colors

    @staticmethod
    def create_data_files(colors, ascii_art):
        with open('colors.txt', 'w') as f:
            color = [str(c) for c in colors]
            f.write(' '.join(color))
        with open('ascii.txt', "w") as f:
            f.write(''.join(ascii_art))

    @staticmethod
    def create_color_array(path):
        with open(path, 'r') as f:
            c = f.read()
            color = c.split(' ')
            colors = [(color[i * 3], color[i * 3 + 1], color[i * 3 + 2]) for i in range(int(len(color) / 3))]
            color = [[int(c.replace(',', '').replace('(', '').replace(')', '')) for c in color] for color in colors]
            colors = [(i[0], i[1], i[2]) for i in color]
            return colors

    @staticmethod
    def ascii_to_image(ascii_art, colors, output_path):
        lines = ascii_art.split("\n")[:-1]  # Exclude last empty line
        width = len(lines[0])
        height = len(lines)

        # Create a new image with RGB mode
        image = Image.new("RGB", (width, height), color="white")
        pixels = image.load()

        for y in range(height):
            for x in range(width):
                # Get ASCII character representing intensity
                char = lines[y][x]
                # Calculate grayscale value based on ASCII character
                char_index = ASCII_CHARS.index(char)
                grayscale = int(char_index / (len(ASCII_CHARS) - 1) * 255)
                # Set pixel color based on grayscale value
                pixels[x, y] = colors[y * width + x]

        image.save(output_path)