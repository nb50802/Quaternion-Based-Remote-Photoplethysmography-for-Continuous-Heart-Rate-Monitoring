import cv2
import numpy as np
from collections import deque

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def apply_signal_processing(intensity):
  # Perform signal processing on the intensity
  # Example: Apply a low-pass filter
  filtered_intensity = apply_low_pass_filter(intensity)
  return filtered_intensity

def apply_fft(intensity):
  # Apply Fast Fourier Transform (FFT) to the intensity
  fft_intensity = np.fft.fft(intensity)
  return fft_intensity

def estimate_heart_rate(avg_red_intensity, avg_green_intensity, avg_blue_intensity):
  # Implement your heart rate estimation logic here
  # Example: Using a simple formula based on RGB intensities
  heart_rate = avg_red_intensity * 0.33 + avg_green_intensity * 0.34 + avg_blue_intensity * 0.33
  return heart_rate

def calculate_heart_rate(avg_red_intensity, avg_green_intensity, avg_blue_intensity):
  # Ensure input intensities are real numbers
  avg_red_intensity = avg_red_intensity.real
  avg_green_intensity = avg_green_intensity.real
  avg_blue_intensity = avg_blue_intensity.real
  # Calculate heart rate based on RGB intensities
  heart_rate = estimate_heart_rate(avg_red_intensity, avg_green_intensity, avg_blue_intensity)
  return heart_rate

def analyze_region(frame, region):
  x, y, w, h = region
  region = frame[y:y + h, x:x + w]
  avg_red = np.mean(region[:, :, 2])
  avg_green = np.mean(region[:, :, 1])
  avg_blue = np.mean(region[:, :, 0])
  return avg_red, avg_green, avg_blue

def apply_low_pass_filter(intensity):
  # Implement your low-pass filter logic here
  # Example: Using a simple moving average filter
  window_size = 5
  filtered_intensity = np.convolve(intensity, np.ones(window_size) / window_size, mode='same')
  return filtered_intensity

def calculate_avg_intensities(frame, forehead_region, left_cheek_region, right_cheek_region):
  # Analyze each region
  avg_red_forehead, avg_green_forehead, avg_blue_forehead = analyze_region(frame, forehead_region)
  avg_red_left_cheek, avg_green_left_cheek, avg_blue_left_cheek = analyze_region(frame, left_cheek_region)
  avg_red_right_cheek, avg_green_right_cheek, avg_blue_right_cheek = analyze_region(frame, right_cheek_region)
  # Calculate the average intensity across all regions
  avg_red_intensity = (avg_red_forehead + avg_red_left_cheek + avg_red_right_cheek) / 3
  avg_green_intensity = (avg_green_forehead + avg_green_left_cheek + avg_green_right_cheek) / 3
  avg_blue_intensity = (avg_blue_forehead + avg_blue_left_cheek + avg_blue_right_cheek) / 3
  return avg_red_intensity, avg_green_intensity, avg_blue_intensity

def calculate_quan_intensities(frame, forehead_region, left_cheek_region, right_cheek_region):
  # Analyze each region
  avg_red_forehead, avg_green_forehead, avg_blue_forehead = analyze_region(frame, forehead_region)
  avg_red_left_cheek, avg_green_left_cheek, avg_blue_left_cheek = analyze_region(frame, left_cheek_region)
  avg_red_right_cheek, avg_green_right_cheek, avg_blue_right_cheek = analyze_region(frame, right_cheek_region)
  # Calculate the squared values for each region
  quan_red_forehead = avg_red_forehead ** 2
  quan_green_forehead = avg_green_forehead ** 2
  quan_blue_forehead = avg_blue_forehead ** 2
  quan_red_left_cheek = avg_red_left_cheek ** 2
  quan_green_left_cheek = avg_green_left_cheek ** 2
  quan_blue_left_cheek = avg_blue_left_cheek ** 2
  quan_red_right_cheek = avg_red_right_cheek ** 2
  quan_green_right_cheek = avg_green_right_cheek ** 2
  quan_blue_right_cheek = avg_blue_right_cheek ** 2
  # Calculate the square root of the sum of squared values for each intensity
  quan_red_intensity = np.sqrt(quan_red_forehead + quan_red_left_cheek + quan_red_right_cheek)
  quan_green_intensity = np.sqrt(quan_green_forehead + quan_green_left_cheek + quan_green_right_cheek)
  quan_blue_intensity = np.sqrt(quan_blue_forehead + quan_blue_left_cheek + quan_blue_right_cheek)
  return quan_red_intensity, quan_green_intensity, quan_blue_intensity

def main():
  # Initialize variables
  previous_avg_red_intensity = previous_avg_green_intensity = previous_avg_blue_intensity = None
  previous_quan_red_intensity = previous_quan_green_intensity = previous_quan_blue_intensity = None
  # Initialize deques to store heart rate values over time
  avg_heart_rates = deque(maxlen=60)
  quan_heart_rates = deque(maxlen=60)
  # Open video capture
  cap = cv2.VideoCapture(0)
  alpha_rgb = 0.08
  alpha = 1 - alpha_rgb
  while True:
  # Read frame from the video capture
  ret, frame = cap.read()
  # Perform face detection and region extraction
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  for (x, y, w, h) in faces:
  # Define regions based on detected face
  forehead_region = (x + int(w * 0.25), y + int(h * 0.05), int(w * 0.5), int(h * 0.20))
  left_cheek_region = (x + int(w * 0.19), y + int(h * 0.5), int(min(w, h) * 0.20), int(min(w, h) * 0.20))
  right_cheek_region = (
  x + w - int(min(w, h) * 0.25) - int(w * 0.15), y + int(h * 0.5), int(min(w, h) * 0.20),
  int(min(w, h) * 0.20))
  # Calculate average RGB intensities
  avg_red_intensity, avg_green_intensity, avg_blue_intensity = calculate_avg_intensities(frame, forehead_region,
  left_cheek_region, right_cheek_region)
  if previous_avg_red_intensity is not None:
  avg_red_intensity = (avg_red_intensity ** alpha) + (previous_avg_red_intensity ** alpha)
  if previous_avg_green_intensity is not None:
  avg_green_intensity = (avg_green_intensity ** alpha) + (previous_avg_green_intensity ** alpha)
  if previous_avg_blue_intensity is not None:
  avg_blue_intensity = (avg_blue_intensity ** alpha) + (previous_avg_blue_intensity ** alpha)
  # Calculate quan RGB intensities
  quan_red_intensity, quan_green_intensity, quan_blue_intensity = calculate_quan_intensities(frame, forehead_region,
  left_cheek_region, right_cheek_region)
  # Apply EMA smoothing for the quan RGB intensities
  if previous_quan_red_intensity is not None:
  quan_red_intensity = (quan_red_intensity ** alpha) + (previous_quan_red_intensity ** alpha)
  if previous_quan_green_intensity is not None:
  quan_green_intensity = (quan_green_intensity ** alpha) + (previous_quan_green_intensity ** alpha)
  if previous_quan_blue_intensity is not None:
  quan_blue_intensity = (quan_blue_intensity ** alpha) + (previous_quan_blue_intensity ** alpha)
  # Apply signal processing to averaged RGB intensities
  filtered_avg_red_intensity = apply_signal_processing(avg_red_intensity)
  filtered_avg_green_intensity = apply_signal_processing(avg_green_intensity)
  filtered_avg_blue_intensity = apply_signal_processing(avg_blue_intensity)
  # Apply FFT to the filtered averaged RGB intensities
  fft_avg_red_intensity = apply_fft(filtered_avg_red_intensity)
  fft_avg_green_intensity = apply_fft(filtered_avg_green_intensity)
  fft_avg_blue_intensity = apply_fft(filtered_avg_blue_intensity)
  # Apply signal processing to quan RGB intensities
  filtered_quan_red_intensity = apply_signal_processing(quan_red_intensity)
  filtered_quan_green_intensity = apply_signal_processing(quan_green_intensity)
  filtered_quan_blue_intensity = apply_signal_processing(quan_blue_intensity)
  # Apply FFT to the filtered quan RGB intensities
  fft_quan_red_intensity = apply_fft(filtered_quan_red_intensity)
  fft_quan_green_intensity = apply_fft(filtered_quan_green_intensity)
  fft_quan_blue_intensity = apply_fft(filtered_quan_blue_intensity)
  # Update the previous intensities
  previous_avg_red_intensity = avg_red_intensity
  previous_avg_green_intensity = avg_green_intensity
  previous_avg_blue_intensity = avg_blue_intensity
  previous_quan_red_intensity = quan_red_intensity
  previous_quan_green_intensity = quan_green_intensity
  previous_quan_blue_intensity = quan_blue_intensity
  # Calculate heart rates
  avg_heart_rate = calculate_heart_rate(fft_avg_red_intensity, fft_avg_green_intensity, fft_avg_blue_intensity)
  quan_heart_rate = calculate_heart_rate(fft_quan_red_intensity, fft_quan_green_intensity, fft_quan_blue_intensity)
  # Append heart rate values to the deques
  avg_heart_rates.append(avg_heart_rate)
  quan_heart_rates.append(quan_heart_rate)
  # Draw rectangles for visualization
  cv2.rectangle(frame, forehead_region[:2], (forehead_region[0] + forehead_region[2], forehead_region[1] + forehead_region[3]),
  (255, 255, 0), 2)
  left_cheek_region[3]), (255, 255, 0), 2)
  cv2.rectangle(frame, left_cheek_region[:2], (left_cheek_region[0] + left_cheek_region[2], left_cheek_region[1] +
  cv2.rectangle(frame, right_cheek_region[:2], (right_cheek_region[0] + right_cheek_region[2], right_cheek_region[1] +
  right_cheek_region[3]), (255, 255, 0), 2)
  # Display heart rates on the frame
  cv2.putText(frame, f"Quan HR: {np.mean(quan_heart_rates):.2f} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 10) #
  Black border
  cv2.putText(frame, f"Quan HR: {np.mean(quan_heart_rates):.2f} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #
  Green Text
  # Display the frame
  cv2.imshow('Frame', frame)
  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
  break
  # Release the video capture and close windows
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
