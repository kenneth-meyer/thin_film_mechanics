"""
    Sets up and runs a boundary value problem (BVP) for a substrate-only system.

    This is going to be used to test my ability to deform the surface of a substrate in a sinusoidal manner,
    my main concern is trying to determine what the element size needs to be to capture wrinkling (if it even matters)

    This also serves as a good first example of a sample workflow that uses CARDIAX.

    NOTE: `cardiax.input_file_handler.py` might need to be updated as:
        1. I don't think it's fully functional
        2. I'm using non-cardiax based classes that might be better off staying outside of cardiax for now
"""

