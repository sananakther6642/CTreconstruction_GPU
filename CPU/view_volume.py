import h5py
import matplotlib.pyplot as plt

# Load the volume
with h5py.File('reconstruction_result_cpp.hdf5', 'r') as f:
    volume = f['Volume'][:]

# Setup the interactive plot
fig, ax = plt.subplots()
slice_idx = volume.shape[0] // 2
im = ax.imshow(volume[slice_idx, :, :], cmap='gray')
ax.set_title(f'Slice {slice_idx}/{volume.shape[0]} (Use Up/Down or Scroll)')

def update_slice(event):
    global slice_idx
    if event.name == 'scroll_event':
        slice_idx += 1 if event.step > 0 else -1
    elif event.name == 'key_press_event':
        if event.key == 'up':
            slice_idx += 1
        elif event.key == 'down':
            slice_idx -= 1
            
    # Keep slice index within bounds
    slice_idx = max(0, min(volume.shape[0] - 1, slice_idx))
    
    im.set_data(volume[slice_idx, :, :])
    ax.set_title(f'Slice {slice_idx}/{volume.shape[0]}')
    fig.canvas.draw_idle()

# Connect mouse scroll and keyboard arrow keys
fig.canvas.mpl_connect('scroll_event', update_slice)
fig.canvas.mpl_connect('key_press_event', update_slice)

plt.show()
