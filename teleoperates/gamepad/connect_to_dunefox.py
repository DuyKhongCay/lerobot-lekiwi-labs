import hid
import time
import sys

# Constants for DuneFox Dongle
TARGET_VID = 0x04B5
TARGET_PID = 0x2413

# Define the default idle packet sent by the gamepad when no inputs are made
IDLE_STATE = [128, 128, 128, 128, 8, 0, 0, 0, 0]

def connect_to_dunefox():
    # Enumerate all devices matching the DuneFox VID and PID
    devices = hid.enumerate(TARGET_VID, TARGET_PID)
    
    if not devices:
        print("Error: DuneFox Dongle not found. Please check connection.")
        sys.exit(1)
        
    # Sort devices by path to ensure we connect to the primary interface (e.g., ...:1.0)
    devices.sort(key=lambda x: x['path'])
    target_path = devices[0]['path']
    
    try:
        # Initialize the HID device object
        gamepad = hid.device()
        
        # Open connection using the exact device path
        print(f"Attempting to open device at path: {target_path}")
        gamepad.open_path(target_path)
        
        # Set non-blocking mode to 1 so the read loop doesn't freeze the program
        gamepad.set_nonblocking(1)
        
        print("Connection successful! Listening for input...")
        print("Press any button or move the joystick. Press Ctrl+C to exit.\n")
        
        # Infinite loop to poll device data
        while True:
            # Read up to 64 bytes of data from the gamepad
            report = gamepad.read(64)
            
            # Filter the report: process only if data exists AND is not the idle state
            if report and report != IDLE_STATE:
                print(f"Action detected: {report}")
                
            # Maintain a short delay to match the polling rate and save CPU cycles
            time.sleep(0.01)
            
    except IOError as e:
        print(f"Permission Error: {e}")
        print("Please ensure udev rules are applied and Dongle is replugged.")
    finally:
        # Ensure the device connection is properly closed when exiting
        gamepad.close()
        print("Connection closed.")

if __name__ == "__main__":
    connect_to_dunefox()