---
layout: post
title:  "Simulating a Red 25 Optical Filter"
---

I wanted to dabble around with trying to "emulate" a physical optic filter in Python for fun. An optical filter is a material placed before a lens that rejects certain wavelengths or energies of light. For example, a green filter rejects much of red and blue wavelengths, only letting green pass through. 

I'm using the good old Bliss wallpaper of Windows XP for this\\
![Windows XP Bliss](/assets/images/bliss.png)

A pretty crude and first-pass implementation of such a filter would be something like this:
```python

import cv2
import numpy as np

img = cv2.imread("bliss.png")

red_matrix = np.array([
    [0, 0, 0],
    [0.01, 0.01, 0.01],
    [0.05, 0.01, 0.9]
])
img_filtered = cv2.transform(img, red_matrix)

cv2.imshow("Crude red filter", img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
The resulting image looks like this\\
![Windows XP Bliss passed through a crude red filter](/assets/images/bliss_redfilter_crude.png)

This already looks decent. Although the first thought that comes up is to merely weight the 3 colour channels accordingly, it's not exactly how it works. A pixel on the image sensor would still see all three
colour components regardless, just in different proportions. The first row of the matrix `red_matrix` simulates the BGR proportion that blue (B) pixels of the sensor would see. The second row simulates the proportions as seen by a green (G) pixel. The third row corresponds to the same for a red (R) pixel. Here, the red pixels would see a little bit of blue and green but they're energized more with red wavelengths.

This is naturally not quite accurate realistically speaking, because we're working in three discrete wavelengths or colour components. Real-life filters and sensors engage with the complete continuous spectrum of visible-light wavelengths (actually more but let's limit the scope to visible light given the topic). At the very least, we can attempt to bring the code closer to that goalpost if not exactly in the goalpost.

The [Wratten Red 25 filter](https://www.edmundoptics.in/p/red-25-kodak-wratten-color-filter/10790/) is a good one to model. 