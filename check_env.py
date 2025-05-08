# check_env.py
import sys
import os
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

print("\nAttempting to import rtlsdr...")
try:
    import rtlsdr
    print(f"Successfully imported rtlsdr module.")
    print(f"rtlsdr module location: {rtlsdr.__file__}")

    print("\nAttempting to instantiate RtlSdr (this tests librtlsdr linking)...")
    # This instantiation is a key test: it checks if pyrtlsdr can find and use the underlying librtlsdr C library.
    sdr = rtlsdr.RtlSdr() 
    print("Successfully instantiated RtlSdr.")
    # Clean up by closing the device if it was opened.
    try:
        sdr.close()
    except Exception as close_err:
        print(f"(Ignoring error during sdr.close() for this test: {close_err})")
    del sdr # Remove the sdr object
    print("Test complete: rtlsdr module imported and RtlSdr class instantiated successfully.")

except ImportError as e:
    print(f"ImportError for rtlsdr: {e}")
    if "No module named 'rtlsdr'" in str(e):
        print("This specific error indicates Python's import system cannot find the pyrtlsdr library files.")
        print("Check if the path printed above for 'rtlsdr module location' (if import succeeded partially) is correct and accessible.")
    else:
        # This could be an error from within pyrtlsdr, often related to librtlsdr
        print("This ImportError might indicate an issue with a dependency of pyrtlsdr or, more likely, the underlying librtlsdr C library.")
        print("Common causes: librtlsdr.dylib not found by the system's dynamic linker, or an architecture mismatch (e.g., ARM Python with Intel librtlsdr).")
except Exception as e:
    # Catches other errors, especially those from the RtlSdr() instantiation like OSError if librtlsdr.dylib is not found
    print(f"A non-ImportError occurred, often related to librtlsdr: {e}")
    print("This frequently points to an issue with the librtlsdr C library itself (e.g., not found by the dynamic linker, permissions, or architecture mismatch).")
    print("Ensure librtlsdr is correctly installed and accessible in your system's library paths (e.g., /usr/local/lib or /opt/homebrew/lib).")