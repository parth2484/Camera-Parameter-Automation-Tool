📷 Camera Parameter Automation & Logging Tool

A Python based tool for real-time camera control, parameter standardization, and ICV workflow support.

This project provides a production-aligned camera control module designed for environments that require consistent, reproducible, and well-documented camera settings such as ICV (In-Camera Verification), automated inspection, and AI-assisted quality systems.

It allows engineers to tune exposure, brightness, and contrast in real time, while automatically logging every parameter change for traceability. Annotated snapshots enable reliable documentation for calibration workflows and AI model inputs.

🚀 Key Features
🔧 Real-Time Camera Control

Live adjustment of exposure, brightness, and contrast

Clean UI sliders + optional numeric input

Real-time preview window with parameter overlays

📝 Automated Parameter Logging

Every change logged to CSV with a timestamp

Enables reproducibility, traceability, and standardized documentation

Ideal for production environments requiring strict calibration records

📸 Annotated Snapshot Capture

Capture snapshots with parameters printed on the frame

Useful for AI datasets, camera alignment, and system optimization

🖥️ Simulation Mode

Run without a physical camera

Generates synthetic frames for testing or demonstration

🧩 Why This Tool Matters

Industrial environments especially automotive OEMs depend on consistent camera behavior for inspection and AI workflows.
This tool helps achieve:

Standardized camera modules

Repeatable calibration processes

Documented changes for audits and reviews

It serves as a foundational component before integrating with advanced ICV or AI-based defect detection systems.

📦 Installation

pip install opencv-python numpy

▶️ Usage
Basic usage
python camera_control.py

Simulation mode (no camera needed)
python camera_control.py --simulate

Custom log and snapshot directories
python camera_control.py --log logs/log.csv --save-dir snapshots/


Keyboard Controls:

q → Quit

s → Save snapshot

r → Reset parameters

Arrow Keys → Adjust brightness/contrast

🗂️ Project Structure
camera_control.py
logs/
snapshots/
README.md

📘 Ideal For

Students applying to production planning, ICV, or AI inspection internships

Engineers needing a reproducible camera-tuning workflow

Teams standardizing camera configurations across multiple stations

⭐ Future Enhancements

Hardware SDK integration (Basler, FLIR, IDS)

Auto-exposure analysis module

AI-assisted optimal parameter selection
