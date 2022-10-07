import subprocess

for flex_rate in [450,500,550,600,650,700,750,800]:
    for ext_rate in [400]:
        print (subprocess.run(["python3", "simulator.py", str(flex_rate), str(ext_rate)], check=True))

# Now read from generated CSVs
