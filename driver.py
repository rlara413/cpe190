from ubidots import ApiClient
import RPi.GPIO as GPIO 
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(7, GPIO.IN)

peoplecount = 0
counter = 0
try: 
    api = ApiClient(token='BBFF-jvBXecob5OfXxnHc0Rh5UZP0Nfjz4a')
    people = api.get_variable('61905080d467923fefef3af3')
except: 
    print ("Couldn't connect to the API, check your Internet connection")
    counter = 0
    peoplev = 0

while (1): 
    presence = GPIO.input(7) 
    if (presence): 
        peoplecount += 1 
        presence = 0
        time.sleep(1.5)
        time.sleep(1)
        counter += 1

    if (counter == 10):
        print (peoplecount) 
        people.save_value({'value':peoplecount})
        counter = 0
        peoplev = 0
