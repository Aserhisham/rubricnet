import os
try:
    import guitarpro
    print("PyGuitarPro is installed.")
except ImportError:
    print("PyGuitarPro NOT installed.")

try:
    import music21
    print("Music21 is installed.")
except ImportError:
    print("Music21 NOT installed.")
