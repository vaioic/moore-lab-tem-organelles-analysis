from pathlib import Path

import hyperspy.api as hs
import matplotlib.pyplot as plt

print(Path.cwd())


# Load the dm4 file
s = hs.load(Path("../data/Dataset01/raw/Mouse192/1/1.dm4"))

# If the file contains a single dataset/image:
data = s.data  # Returns a NumPy array
print(data.shape)

# If you want to plot or inspect it:
plt.imshow(data)
plt.show()
