import hid
import time
import sys

# Constants for DuneFox Dongle
TARGET_VID = 0x04B5
TARGET_PID = 0x2413

# The default idle packet sent by the gamepad
IDLE_STATE = [128, 128, 128, 128, 8, 0, 0, 0, 0]

def parse_gamepad_data(report):
    """
    Parses the raw 9-byte report array into a readable dictionary.
    """
    # Return empty state if report is invalid or too short
    if not report or len(report) < 9:
        return None

    # Parse D-pad directions based on byte 4
    dpad_val = report[4]
    
    # Initialize the parsed state dictionary
    state = {
        'joysticks': {
            'left_x': report[0],
            'left_y': report[1],
            'right_x': report[2],
            'right_y': report[3]
        },
        'triggers_analog': {
            'lt': report[7],
            'rt': report[8]
        },
        'dpad': {
            'up': dpad_val == 0,
            'down': dpad_val == 4,
            'left': dpad_val == 2,
            'right': dpad_val == 6,
            'idle': dpad_val == 8
        },
        'buttons': {
            # Use bitwise AND (&) to detect simultaneous button presses on byte 6
            'a': bool(report[6] & 128),
            'b': bool(report[6] & 64),
            'x': bool(report[6] & 16),
            'y': bool(report[6] & 8),
            'lb': bool(report[6] & 2),
            'rb': bool(report[6] & 1),
            
            # Check byte 5 for digital trigger clicks
            'lt_click': bool(report[5] & 128),
            'rt_click': bool(report[5] & 64)
        }
    }
    return state

def connect_to_dunefox():
    # Enumerate devices matching the DuneFox VID and PID
    devices = hid.enumerate(TARGET_VID, TARGET_PID)
    
    if not devices:
        print("Error: DuneFox Dongle not found.")
        sys.exit(1)
        
    # Sort by path to connect to the primary interface
    devices.sort(key=lambda x: x['path'])
    target_path = devices[0]['path']
    
    try:
        # Initialize and open the HID device
        gamepad = hid.device()
        gamepad.open_path(target_path)
        gamepad.set_nonblocking(1) # Non-blocking mode
        
        print("Connection successful! Listening for input. Press Ctrl+C to exit.\n")
        
        # Read loop
        while True:
            report = gamepad.read(64)
            
            # Process only if data exists and is not the idle state
            if report and report != IDLE_STATE:
                parsed_state = parse_gamepad_data(report)
                
                # Print active buttons for demonstration
                active_buttons = [btn for btn, pressed in parsed_state['buttons'].items() if pressed]
                active_dpad = [dir for dir, pressed in parsed_state['dpad'].items() if pressed and dir != 'idle']
                
                print(f"Buttons: {active_buttons} | D-Pad: {active_dpad} | " 
                      f"LT: {parsed_state['triggers_analog']['lt']} | "
                      f"RT: {parsed_state['triggers_analog']['rt']} | "
                      f"LX: {parsed_state['joysticks']['left_x']} LY: {parsed_state['joysticks']['left_y']}"
                      f" RX: {parsed_state['joysticks']['right_x']} RY: {parsed_state['joysticks']['right_y']}")
                
            # Sleep to match polling rate and save CPU
            time.sleep(0.01)
            
    except IOError as e:
        print(f"Error: {e}")
    finally:
        # Close connection gracefully
        gamepad.close()

if __name__ == "__main__":
    connect_to_dunefox()