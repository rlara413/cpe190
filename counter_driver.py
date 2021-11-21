import RPi.GPIO as GPIO 
import time

GPIO.setmode(GPIO.BCM)
sensor_in = 11
sensor_out = 9

GPIO.setup(sensor_in, GPIO.IN)
GPIO.setup(sensor_out, GPIO.IN)

counter = 0
capacity = 20

# Driver here
while (1):
    if GPIO.input(sensor_in) and not GPIO.input(sensor_out) and counter < capacity:
        print("New Person Detected!")
        counter += 1;
        print("TOTAL:")
        print(counter)
        print("\n")
        tempsense()

    if GPIO.input(sensor_out) and not GPIO.input(sensor_in) and counter > 0:
        print("Person Left!")
        counter -= 1;
        print("TOTAL:")
        print(counter)
        print("\n")
    if capacity <= counter:
        print("capacity met!")

    time.sleep(0.25)

# Check Temperature of Person
def tempsense():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    print ("Person Temperature :"), sensor.get_object_1()
    temp = sensor.get_object_1()
    bus.close()
    if temp > 37:
        print("temperature too high")
    else:
        unlock()

# Lock System
def unlock():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    print("door unlocked please enter before the door locks again \n")
    GPIO.output(17, GPIO.HIGH)
    time.sleep(5)
    GPIO.output(17, GPIO.LOW)
    time.sleep(2)
