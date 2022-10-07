import subprocess

for motor_rate in [50,100,150,200,250,300,350,400,450,500]:
    print (subprocess.run(["python3", "simulator_motor.py",str(0.0), str(0.0), str(0.0),str(motor_rate),str(motor_rate)], check=True))

# Now read from generated CSVs
