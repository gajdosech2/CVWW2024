import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import time
import random
import tensorflow as tf
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.BinaryIoU()]

#from google.colab import drive
#drive.mount('/content/gdrive')
root_path = './'

MOTION_CAM_MAX_RANGE = 1800
#DATASET_THRUSTER = root_path + '/Skeletex3DData/thruster/'
DATASET_ARMADILLO = root_path + '/Armadillo/'
MODELS = root_path + '/Models/'
DATASET = DATASET_ARMADILLO
NOISE_MULT = 2.0

"""# Data Loading

"""

def load_scan(positions_path, normals_path, load_xyz=False):
  positions = cv2.imread(positions_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  positions = cv2.resize(positions, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
  normals = cv2.imread(normals_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  normals = cv2.resize((normals + 1) / 2, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
  xyz = []
  if (load_xyz):
    for y in range(positions.shape[0]):
      for x in range(positions.shape[1]):
        if random.randrange(20) == 0 and not (np.all(positions[y][x] == 0)):
          xyz.append(positions[y][x])
    xyz = np.array(xyz)
  return positions, normals, xyz

p_path = DATASET + 'train/scan_003_positions.exr'
n_path = DATASET + 'train/scan_003_normals.exr'
positions, normals, xyz = load_scan(p_path, n_path, True)
print(positions.shape)
print(normals.shape)
print(xyz.shape)

trace1 = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=-xyz[:, 2], mode='markers',
                      marker=dict(size=2, color=xyz[:, 2], colorscale='Viridis', opacity=0.7))

data = [trace1]

layout = go.Layout(height=800, width=800, title="Bin Scan")
fig = go.Figure(data=data, layout=layout)
fig.show()

def load_mask(labels_path):
  mask = cv2.imread(labels_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
  mask = mask[:, :, 2]
  mask = np.expand_dims(mask, axis=2)
  mask = (mask > 1).astype(float)
  return mask

def show(depth, normals, mask):
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 3, 1)
  plt.axis('off')
  plt.imshow(-depth).set_cmap('gray')

  plt.subplot(1, 3, 2)
  plt.axis('off')
  plt.imshow(normals)

  plt.subplot(1, 3, 3)
  plt.axis('off')
  plt.imshow(mask[:, :, 0]).set_cmap('gray')

  plt.savefig('show.png')
  plt.show()

mask = load_mask(DATASET + '/train/scan_003_labels.png')
print(np.amax(mask))
print(mask.shape)
show(positions[:, :, 2], normals, mask)

"""# Dataset Parser"""

def axial_noise_polynomial(z, o):
  return 0.599 - 1.43 * 10**(-3) * z - 8.94 * 10**(-3) * o + 8.84 * 10**(-7) * z**(2) + 1.27 * 10**(-5) * z * o + 2.75 * 10**(-5) * o**(2)

def lateral_noise_polynomial(z, o):
  return 0.915 - 6.91 * 10**(-5) * z + 2.84 * 10 ** (-3) * o

def generate_axial_noise(depth, angles):
  sigma_values = np.vectorize(axial_noise_polynomial)(depth, angles)
  noise = np.random.normal(loc=0, scale=sigma_values)  
  return noise

# def generate_axial_noise_loop(depth, angles):
#   noise = np.zeros((depth.shape[0], depth.shape[1]))
#   for y in range(noise.shape[0]):
#     for x in range(noise.shape[1]):
#       sigma = axial_noise_polynomial(depth[y][x], angles[y][x])
#       noise[y][x] = np.random.normal(loc=0, scale=sigma)
#   return np.array(noise)

def generate_lateral_noise(depth, angles):
  sigma_values = np.vectorize(lateral_noise_polynomial)(depth, angles)
  noise_dx = np.random.normal(loc=0, scale=sigma_values)  
  noise_dy = np.random.normal(loc=0, scale=sigma_values)  
  noise = np.concatenate([np.expand_dims(noise_dx, axis=2), np.expand_dims(noise_dy, axis=2)], axis=2)
  return noise

# def generate_lateral_noise_loop(depth, angles):
#   noise = np.zeros((depth.shape[0], depth.shape[1], 2))
#   for y in range(noise.shape[0]):
#     for x in range(noise.shape[1]):
#       sigma = lateral_noise_polynomial(depth[y][x], angles[y][x])
#       noise[y][x][0] = np.random.normal(loc=0, scale=sigma)
#       noise[y][x][1] = np.random.normal(loc=0, scale=sigma)
#   return np.array(noise)

def add_axial_noise(depth, angles):
  return depth + generate_axial_noise(depth, angles) * NOISE_MULT

def add_lateral_noise(stacked, depth, angles):
  noisy = stacked.copy()
  noise = generate_lateral_noise(depth, angles) * NOISE_MULT
  y, x = np.indices(depth.shape)
  dy, dx = noise[:,:,0], noise[:,:,1]
  newY, newX = np.round(y + dy).astype(int), np.round(x + dx).astype(int)

  newY = np.clip(newY, 0, noisy.shape[0] - 1)
  newX = np.clip(newX, 0, noisy.shape[1] - 1)
  noisy[newY, newX] = stacked[y, x]

  return noisy

# def add_lateral_noise_loop(map, depth, angles):
#   noisy = map.copy()
#   noise = generate_lateral_noise(depth, angles) * NOISE_MULT

#   for y in range(noisy.shape[0]):
#     for x in range(noisy.shape[1]):
#       dy, dx = noise[y][x]
#       newY = int(np.round(y + dy))
#       newX = int(np.round(x + dx))
#       if 0 <= newY < noisy.shape[0] and 0 <= newX < noisy.shape[1]:
#         noisy[newY, newX] = map[y, x]

#   return np.array(noisy)

class Generator(tf.keras.utils.Sequence):
    def __init__(self, datasets_path, gt=True, noise=True):
      self.masks, self.positions, self.inputs, self.xyzs = [], [], [], []
      self.gt = gt
      self.noise = noise
      self.load_data(datasets_path)

    def load_data(self, dataset_path):
      files = os.listdir(dataset_path)
      files.sort()
      #random.shuffle(files)
      for f in files:
        if "_positions" in f:
          print(dataset_path + f[:-13])
          p_path = dataset_path + f
          n_path = dataset_path + f[:-13] + 'normals.exr'
          positions, normals, xyz = load_scan(p_path, n_path, not self.gt)

          depth = positions[:, :, 2]
          normals = normals
          angles = np.zeros(depth.shape)

          if NOISE_MULT > 0 and self.noise:
            depth = add_axial_noise(depth, angles)

          normalized = depth / MOTION_CAM_MAX_RANGE
          normalized = np.expand_dims(normalized, axis=2)
          stacked = np.concatenate([normals, normalized], axis=2)

          if NOISE_MULT > 0 and self.noise:
            stacked = add_lateral_noise(stacked, depth, angles)

          self.inputs.append(np.expand_dims(stacked, axis=0))
          if self.gt:
            mask = load_mask(dataset_path + f[:-13] + 'labels.png')
            self.masks.append(np.expand_dims(mask, axis=0))
          else:
            self.positions.append(positions)
            self.xyzs.append(xyz)

    def get_positions_and_xyzs(self, index):
      return self.positions[index], self.xyzs[index]

    def __len__(self):
      return len(self.masks)

    def __getitem__(self, index):
      return self.inputs[index], self.masks[index] if self.gt else None

train_generator = Generator(DATASET + 'train/')
val_generator = Generator(DATASET + 'synth-val/')
test_generator = Generator(DATASET + 'synth-test/')
real_generator = Generator(DATASET + 'real-val/', gt=False, noise=False)
real_gt = Generator(DATASET + 'real-val/', gt=True, noise=False)

train_input, train_mask = train_generator[0]
show(train_input[0, :, :, 3], train_input[0, :, :, :3], train_input[0, :, :, :3])

print(train_input.shape)
print(train_mask.shape)

"""# Neural Network

"""

def net(i):
    c1 = tf.keras.layers.Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (i)
    c1 = tf.keras.layers.Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (c1)
    p1 = tf.keras.layers.MaxPooling2D(2) (c1)

    c2 = tf.keras.layers.Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (p1)
    c2 = tf.keras.layers.Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (c2)
    p2 = tf.keras.layers.MaxPooling2D(2) (c2)

    c3 = tf.keras.layers.Conv2D(64, 5, activation=tf.keras.layers.LeakyReLU(), padding='same') (p2)
    c3 = tf.keras.layers.Conv2D(64, 5, activation=tf.keras.layers.LeakyReLU(), padding='same') (c3)

    u4 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same') (c3)
    u4 = tf.keras.layers.concatenate([u4, c2])
    c4 = tf.keras.layers.Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (u4)
    c4 = tf.keras.layers.Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (c4)

    u5 = tf.keras.layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same') (c4)
    u5 = tf.keras.layers.concatenate([u5, c1])
    c5 = tf.keras.layers.Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (u5)
    c5 = tf.keras.layers.Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(), padding='same') (c5)
    o = tf.keras.layers.Conv2D(1, 1, activation='sigmoid') (c5)
    return o


def generate():
    i = tf.keras.Input(shape=(None, None, 4))
    o = net(i)
    model = tf.keras.Model(inputs=i, outputs=o)

    print(model.summary())
    print('Total number of layers: {}'.format(len(model.layers)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model

model = generate()
#tf.keras.utils.plot_model(model, to_file='model.png')

filepath = 'model.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_binary_io_u',
                             verbose=1,
                             save_best_only=True,
                             mode='max')


start = time.time()
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=48,
                    validation_data=real_gt,
                    validation_steps=len(real_gt),
                    callbacks=[checkpoint])
print(f"Elapsed time: {time.time() - start} seconds")

real_input, _ = real_generator[5]
print(real_input.shape)
prediction = model.predict(real_input)
prediction = (prediction[0] > 0.7).astype(np.uint8)

print(prediction.shape)

kernel = np.ones((5,5),np.uint8)
prediction = cv2.erode(prediction, kernel, iterations = 1)
prediction = cv2.dilate(prediction, kernel, iterations = 4)
prediction = np.expand_dims(prediction, axis=2)

show(real_input[0, :, :, 3], real_input[0, :, :, :3], prediction)

real_val = Generator(DATASET + 'real-val/', gt=True, noise=False)
real_test = Generator(DATASET + 'real-test/', gt=True, noise=False)

model = tf.keras.models.load_model(MODELS + "model_1_25.keras")

for i in [13, 16, 23]:
  test_input, _ = real_test[i]
  prediction = model.predict(test_input)
  prediction = prediction[0]
  show(test_input[0, :, :, 3], test_input[0, :, :, :3], prediction > 0.7)

iou = []
for model_name in ["0_00", "0_25", "0_50", "0_75", "1_00", "1_25", "1_50", "1_75", "2_00", "2_25", "2_50", "2_75", "3_00", "3_25", "3_50", "3_75", "4_00"]:
  model = tf.keras.models.load_model(MODELS + "model_" + model_name + ".keras")
  result = model.evaluate(real_test)
  iou.append(result[4])
  print(model_name)

noises = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4]

color = [{p<0.55: 'red', 0.55<=p<=0.65: 'orange', p>0.65: 'green'}[True] for p in iou]

fig, ax = plt.subplots()
fig.set_figheight(2)
fig.set_figwidth(8)
ax.xaxis.set_ticks(np.arange(0, 4.1, 0.25))
ax.yaxis.set_ticks(np.arange(0.5, 1.0, 0.05))
ax.bar(noises, [(i - 0.45) for i in iou], color =color, width = 0.15, bottom=0.45)
#ax.set(xlabel='Noise Multiplicator', ylabel='IoU on Real Samples',
#       title='Evaluation of Adding Axial and Lateral Noise to Synthetic Data')
ax.grid(which="major")
plt.tight_layout()
#ax.tight_layout()
fig.savefig("noise.pdf")
plt.show()