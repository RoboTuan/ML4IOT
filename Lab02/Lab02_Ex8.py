import tensorflow as tf
import time as t

# Read a 640x480 image
image = tf.io.read_file('./640x480.jpg')

# Convert the image in a TF tensor
image = tf.io.decode_jpeg(image)

# Take a center crop of size 168x168 from the image
center_crop = tf.image.crop_to_bounding_box(image, 112, 224, target_height=168, target_width=168)

# Resize the image to 224x224 with different methods
start_bilinear = t.time()
bilinear = tf.image.resize(center_crop, method='bilinear', size=[224,224])
end_bilinear = t.time()
print(f"Time for bilinear resize method: {end_bilinear-start_bilinear :.3f}")

start_bicubic = t.time()
bicubic = tf.image.resize(center_crop, method='bicubic', size=[224,224])
end_bicubic = t.time()
print(f"Time for bicubic resize method: {end_bicubic-start_bicubic :.3f}")

start_nearest = t.time()
nearest = tf.image.resize(center_crop, method='nearest', size=[224,224])
end_nearest = t.time()
print(f"Time for nearest resize method: {end_nearest-start_nearest :.3f}")

start_area = t.time()
area = tf.image.resize(center_crop, method='area', size=[224,224])
end_area = t.time()
print(f"Time for area resize method: {end_area-start_area :.3f}")


# Store the resized images
bilinear = tf.cast(bilinear, tf.uint8)
bicubic = tf.cast(bicubic, tf.uint8)
nearest = tf.cast(nearest, tf.uint8)
area = tf.cast(area, tf.uint8)

bilinear = tf.io.encode_jpeg(bilinear)
bicubic = tf.io.encode_jpeg(bicubic)
nearest = tf.io.encode_jpeg(nearest)
area = tf.io.encode_jpeg(area)

folder = './images/'
tf.io.write_file(folder+'bilinear.jpg', bilinear)
tf.io.write_file(folder+'bicubic.jpg', bicubic)
tf.io.write_file(folder+'nearest.jpg', nearest)
tf.io.write_file(folder+'area.jpg', area)


