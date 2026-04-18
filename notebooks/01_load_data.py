# BP Lifestyle Impact Project
# Step 1: Load NHANES datasets

import pandas as pd
import pyreadstat

demo, _ = pyreadstat.read_xpt("../data/DEMO_L.xpt")
bp, _   = pyreadstat.read_xpt("../data/BPXO_L.xpt")
bmx, _  = pyreadstat.read_xpt("../data/BMX_L.xpt")
diet, _ = pyreadstat.read_xpt("../data/DR1TOT_L.xpt")
paq, _  = pyreadstat.read_xpt("../data/PAQ_L.xpt")
sleep,_ = pyreadstat.read_xpt("../data/SLQ_L.xpt")

print("All datasets loaded successfully")
