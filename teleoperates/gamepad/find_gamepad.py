import hid

def find_gamepads():
    # Enumerate all connected HID devices
    all_devices = hid.enumerate()
    gamepads = []

    for device in all_devices:
        # usage_page 1 = Generic Desktop
        # usage 5 = Gamepad, usage 4 = Joystick
        #if device['usage_page'] == 1 and device['usage'] in [4, 5]:
        gamepads.append(device)
            
        # Print out clean information for the user
        print(f"--- Gamepad Found ---")
        print(f"Device Name: {device['product_string']}")
        print(f"Manufacturer: {device['manufacturer_string']}")
        print(f"VID: 0x{device['vendor_id']:04X} | PID: 0x{device['product_id']:04X}")
        print(f"Path: {device['path']}\n")    
            
    return gamepads

if __name__ == "__main__":
    gamepads = find_gamepads()
    if not gamepads:
        print("No gamepads detected via HIDAPI.")