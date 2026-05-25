import cv2
import numpy as np
from PIL import Image
import io

def compress_image(image_pil: Image.Image, quality: int):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    return compressed_pil, buffer.getvalue()

def calculate_psnr(orig_cv, comp_cv):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    return psnr(orig_cv, comp_cv, data_range=255)

def get_size_kb(bytes_data):
    return len(bytes_data) / 1024

def create_difference_image(orig_cv, comp_cv):
    """Enhanced difference map - much more visible"""
    diff = cv2.absdiff(orig_cv, comp_cv)
    
    # Convert to grayscale and enhance visibility
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Strong enhancement
    enhanced = cv2.convertScaleAbs(gray_diff, alpha=8, beta=0)   # Increased alpha
    
    # Apply colormap for better visibility (red = more difference)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    
    return colored

def plot_histogram(img_cv, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_cv], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=col.upper())
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt

def resize_image(img_cv, width, height):
    """Resize using OpenCV"""
    return cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_AREA)

def crop_image(img_cv, x, y, w, h):
    """Simple crop function"""
    return img_cv[y:y+h, x:x+w]


def adjust_brightness_contrast(img_cv, brightness=0, contrast=1.0):
    """Basic brightness & contrast adjustment"""
    img = np.int16(img_cv)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)