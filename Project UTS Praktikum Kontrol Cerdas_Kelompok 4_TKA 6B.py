import cv2
import numpy as np
import serial
import time
import threading
import pandas as pd
from datetime import datetime
import re

def nothing(x):
    pass

class SensorDataReader(threading.Thread):
    """
    Threaded serial reader for Arduino sensor data.
    Parses lines with following formats:
    - ULTRA:<value>
    - GYRO:X=<val>,Y=<val>,Z=<val>
    - WIDTH:<value>
    - LENGTH:<value>

    Continuously updates latest sensor values in a thread-safe way.
    """
    def __init__(self, port='COM3', baudrate=9600):
        super().__init__()
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.data_lock = threading.Lock()
        # Store individual sensor values separately
        self.sensor_data = {
            'length': None,
            'width': None,
            'gyro_x': None,
            'gyro_y': None,
            'gyro_z': None,
            'ultrasonic_distance': None
        }

        # Regex patterns for parsing
        self.gyro_pattern = re.compile(r'GYRO:.*?X=([-+]?[0-9]*\.?[0-9]+),\s*Y=([-+]?[0-9]*\.?[0-9]+),\s*Z=([-+]?[0-9]*\.?[0-9]+)', re.IGNORECASE)

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[Sensor] Connected to {self.port} at {self.baudrate} baud.")
        except Exception as e:
            print(f"[Sensor] Could not open serial port {self.port}: {e}")
            return

        self.running = True
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    # Parse ultrasonic sensor value
                    if line.upper().startswith("ULTRA:"):
                        try:
                            val = float(line.split(':')[1].strip())
                            with self.data_lock:
                                self.sensor_data['ultrasonic_distance'] = val
                        except ValueError:
                            pass

                    # Parse gyro values GYRO:X=...,Y=...,Z=...
                    elif line.upper().startswith("GYRO:"):
                        match = self.gyro_pattern.match(line)
                        if match:
                            try:
                                gx = float(match.group(1))
                                gy = float(match.group(2))
                                gz = float(match.group(3))
                                with self.data_lock:
                                    self.sensor_data['gyro_x'] = gx
                                    self.sensor_data['gyro_y'] = gy
                                    self.sensor_data['gyro_z'] = gz
                            except ValueError:
                                pass

                    # Parse width measurement
                    elif line.upper().startswith("WIDTH:"):
                        try:
                            val = float(line.split(':')[1].strip())
                            with self.data_lock:
                                self.sensor_data['width'] = val
                        except ValueError:
                            pass

                    # Parse length measurement
                    elif line.upper().startswith("LENGTH:"):
                        try:
                            val = float(line.split(':')[1].strip())
                            with self.data_lock:
                                self.sensor_data['length'] = val
                        except ValueError:
                            pass

                    # Could add more sensors here if needed
            except Exception as e:
                print(f"[Sensor] Serial read error: {e}")
                break
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Sensor] Serial port closed.")

    def stop(self):
        self.running = False
        self.join()

    def get_data(self):
        with self.data_lock:
            return self.sensor_data.copy()


def main():
    sensor_port = 'COM3'  # Set your Arduino COM port here
    sensor_reader = SensorDataReader(port=sensor_port, baudrate=6900)
    sensor_reader.start()

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or both cameras")
        sensor_reader.stop()
        return

    window_size = 10
    min_disp = 0
    num_disp = 64

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = 63,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Parameter stereo vision
    focal_length = 746.68  # dalam pixel
    baseline = 0.21        # dalam meter

    cv2.namedWindow('Parameters')
    cv2.createTrackbar('CannyThresh1','Parameters',50,300,nothing)
    cv2.createTrackbar('CannyThresh2','Parameters',150,300,nothing)
    cv2.createTrackbar('NumDisp','Parameters',112,160,nothing)

    print("Press ESC to exit, 's' to save current measurement to Excel")

    saved_data = []
    last_measurements = None

    while True:
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()
        if not retL or not retR:
            print("Failed to grab frames")
            break

        frameL = cv2.resize(frameL, (320, 240))
        frameR = cv2.resize(frameR, (320, 240))

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        canny_thresh1 = cv2.getTrackbarPos('CannyThresh1', 'Parameters')
        canny_thresh2 = cv2.getTrackbarPos('CannyThresh2', 'Parameters')
        num_disp = cv2.getTrackbarPos('NumDisp', 'Parameters')
        if num_disp % 16 != 0:
            num_disp = (num_disp //16)*16
        if num_disp < 16:
            num_disp = 16
        stereo.setNumDisparities(num_disp)

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        edges = cv2.Canny(grayL, canny_thresh1, canny_thresh2)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        measurement_text = "No valid measurement"
        length_stereo = None
        width_stereo = None

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) >= 4:
                pts = approx.reshape(-1,2)
                pts = pts[np.argsort(pts[:,0])]
                left_pts = pts[:len(pts)//2]
                right_pts = pts[len(pts)//2:]
                left_bottom = left_pts[np.argmax(left_pts[:,1])]
                left_top = left_pts[np.argmin(left_pts[:,1])]
                right_bottom = right_pts[np.argmax(right_pts[:,1])]
                right_top = right_pts[np.argmin(right_pts[:,1])]
                trapezoid_pts = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)

                points_3D = []
                for (x,y) in trapezoid_pts:
                    x_clamped = np.clip(x, 0, disparity.shape[1]-1)
                    y_clamped = np.clip(y, 0, disparity.shape[0]-1)
                    disp = disparity[y_clamped, x_clamped]
                    if disp <= 0:
                        X = np.nan
                        Y = np.nan
                        Z = np.nan
                    else:
                        Z = (focal_length * baseline) / disp
                        X = (x_clamped - 640/2) * Z / focal_length
                        Y = (y_clamped - 480/2) * Z / focal_length
                    points_3D.append([X,Y,Z])
                points_3D = np.array(points_3D)

                if np.isnan(points_3D).any():
                    measurement_text = "Bad disparity data"
                else:
                    width_bottom = np.linalg.norm(points_3D[0] - points_3D[3])
                    width_top = np.linalg.norm(points_3D[1] - points_3D[2])
                    width_stereo = (width_bottom + width_top)/2

                    length_left = np.linalg.norm(points_3D[0] - points_3D[1])
                    length_right = np.linalg.norm(points_3D[3] - points_3D[2])
                    length_stereo = (length_left + length_right)/2

                    measurement_text = "Stereo W: {:.2f}cm  L: {:.2f}cm".format(width_stereo, length_stereo)

                cv2.polylines(frameL, [trapezoid_pts], isClosed=True, color=(0,255,0), thickness=2)
                for pt in trapezoid_pts:
                    cv2.circle(frameL, tuple(pt), 5, (0,0,255), -1)

                cv2.putText(frameL, measurement_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frameL, "Not enough polygon points", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frameL, "No contour detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        sensor_data = sensor_reader.get_data()

        # Combine stereo and sensor measurements with weights
        if sensor_data['length'] is not None and length_stereo is not None:
            weight_stereo = 0.6
            weight_sensor = 0.4
            combined_length = weight_stereo*length_stereo + weight_sensor*sensor_data['length']
        else:
            combined_length = length_stereo if length_stereo is not None else sensor_data['length']

        if sensor_data['width'] is not None and width_stereo is not None:
            combined_width = weight_stereo*width_stereo + weight_sensor*sensor_data['width']
        else:
            combined_width = width_stereo if width_stereo is not None else sensor_data['width']

        # Print combined measurement in terminal
        print("\rCombined Measurement - Width: {:0.2f} cm, Length: {:0.2f} cm".format(
            combined_width if combined_width else 0,
            combined_length if combined_length else 0
        ), end='')

        # Display sensor info on frame
        if sensor_data['ultrasonic_distance'] is not None:
            sensor_text1 = f"Ultrasonic Dist: {sensor_data['ultrasonic_distance']:.2f} cm"
            cv2.putText(frameL, sensor_text1, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if None not in (sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']):
            sensor_text2 = f"Gyro: X={sensor_data['gyro_x']:.2f}, Y={sensor_data['gyro_y']:.2f}, Z={sensor_data['gyro_z']:.2f}"
            cv2.putText(frameL, sensor_text2, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        combined = cv2.hconcat([frameL, disp_color])

        cv2.imshow('Stereo Vision Railway Measurement', combined)
        cv2.imshow('Canny Edges', edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'): 
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_to_save = {
                'Timestamp': timestamp,
                'Stereo_Width_m': width_stereo if width_stereo is not None else np.nan,
                'Stereo_Length_m': length_stereo if length_stereo is not None else np.nan,
                'Sensor_Width_m': sensor_data['width'] if sensor_data['width'] is not None else np.nan,
                'Sensor_Length_m': sensor_data['length'] if sensor_data['length'] is not None else np.nan,
                'Combined_Width_m': combined_width if combined_width else np.nan,
                'Combined_Length_m': combined_length if combined_length else np.nan,
                'Ultrasonic_m': sensor_data['ultrasonic_distance'] if sensor_data['ultrasonic_distance'] is not None else np.nan,
                'Gyro_X': sensor_data['gyro_x'] if sensor_data['gyro_x'] is not None else np.nan,
                'Gyro_Y': sensor_data['gyro_y'] if sensor_data['gyro_y'] is not None else np.nan,
                'Gyro_Z': sensor_data['gyro_z'] if sensor_data['gyro_z'] is not None else np.nan,
            }
            saved_data.append(data_to_save)
            df = pd.DataFrame(saved_data)
            excel_filename = 'railway_measurements.xlsx'
            try:
                df.to_excel(excel_filename, index=False)
                print(f"\nData saved to {excel_filename}")
            except Exception as e:
                print(f"\nError saving to Excel: {e}")

    cap_left.release()
    cap_right.release()
    sensor_reader.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
