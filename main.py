
import cv2
from skimage import io, color
from analysis.histogram_analysis import analyze_histograms
from analysis.frequency_analysis import compute_fft_magnitude
from analysis.contour_analysis import get_edges
from analysis.texture_analysis import texture_maps
from analysis.noise_analysis import compute_noise_std
from analysis.color_analysis import get_saturation_map
from analysis.statistical_anomaly import median_residual
from analysis.combined_detector import decision_by_thresholds

def analyze_image(image_path):
    image_rgb = io.imread(image_path)
    image_gray = color.rgb2gray(image_rgb)
    image_gray = (image_gray * 255).astype("uint8")

    hists = analyze_histograms(image_rgb)
    fft = compute_fft_magnitude(image_gray)
    sobel, canny = get_edges(image_gray)
    lbp, entropy = texture_maps(image_gray)
    noise_std, residual = compute_noise_std(image_gray)
    saturation = get_saturation_map(image_rgb)
    median = median_residual(image_gray)

    metrics = {
        'noise_std': noise_std,
        'fft_mean': fft.mean(),
        'entropy_mean': entropy.mean(),
        'median_residual_std': median.std()
    }

    thresholds = {
        'noise_std': 5.0,
        'fft_mean': 30.0,
        'entropy_mean': 4.0,
        'median_residual_std': 20.0
    }

    decision = decision_by_thresholds(metrics, thresholds)
    return decision, metrics

if __name__ == "__main__":
    import sys
    decision, metrics = analyze_image(sys.argv[1])
    print("Trigger detected?" if decision else "Image clean.")
    print(metrics)
