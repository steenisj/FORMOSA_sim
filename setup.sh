
echo "Setting PYTHONPATH environment variable..."
export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "Checking for required modules..."

echo -e "from __future__ import print_function\ntry:\n    import numpy\n    print(\"-Successfully loaded numpy\")\n\
except:\n    print(\"ERROR: could not import numpy!\")\n    exit(1)" | python3

echo -e "from __future__ import print_function\ntry:\n    import matplotlib.pyplot\n    print(\"-Successfully loaded matplotlib\")\n\
except:\n    print(\"WARNING: could not import matplotlib!\")\n    exit(1)" | python3

echo -e "from __future__ import print_function\ntry:\n    import ROOT\n    print(\"-Successfully loaded ROOT\")\n\
except:\n    print(\"WARNING: could not import pyROOT!\")\n    exit(1)" | python3


