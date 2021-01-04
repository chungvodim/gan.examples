import glob
import PIL
from PIL import Image
import imageio

# Create a GIF
# Display a single image using the epoch number
EPOCHS=7
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# import tensorflow_docs.vis.embed as embed
# embed.embed_file(anim_file)