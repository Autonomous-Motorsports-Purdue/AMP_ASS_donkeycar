with open(r"parts\zed_calibration_params.bin", 'rb') as f:
    print(f.read(64))  # read first 64 bytes to get a clue