import cv2
import numpy as np

# Create a simple test image
width = 640
height = 480
image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

# Draw some shapes to make it interesting
cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle
cv2.circle(image, (400, 200), 100, (255, 0, 0), -1)  # Blue circle
cv2.putText(image, "Test Image", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)  # Red text

# Save the image
cv2.imwrite('test_image.jpg', image)
print("Test image created successfully!") 