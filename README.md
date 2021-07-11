# Continuous-Thermal-Sensing

This is based on Thermal Face: https://github.com/maxbbraun/thermal-face

Create a virtual environment and install the necessary dependencies:
 - pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtim
 - pip install Pillow
 - pip install opencv-python
 - pip install numpy

I had to install tflite using Anaconda, and thus used an Anaconda virtual environment for this project.


**Inferencing:**
main.py runs a detection on a single image. The image file is givin in the argument --image.

live_detection.py runs a detection on a video feed from the Lepton. To run the program, plug in the Lepton via USB, run the script. Note that you may have to change the --camera argument (for my computer setup, the lepton's camera ID is 1).
