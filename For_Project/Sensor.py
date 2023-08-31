import RPi.GPIO as GPIO
import mfrc522
import signal
import time
GPIO.setwarnings(False)

continue_reading = True

def end_read(signal, frame):
    global continue_reading
    # print ("Ctrl+C captured, ending read.")
    continue_reading = False
    GPIO.cleanup()


# Hook the SIGINT
signal.signal(signal.SIGINT, end_read)

# Create an object of the class MFRC522
MIFAREReader = mfrc522.MFRC522()

# Welcome message
# print ("Welcome to the MFRC522 data read example")
# print ("Press Ctrl-C to stop.")


# This loop keeps checking for chips. If one is near it will get the UID and authenticate
def rfid_state():
    # if continue_reading:
    while continue_reading:

        # Scan for cards
        (status, TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

        # Get the UID of the card
        (status, uid) = MIFAREReader.MFRC522_Anticoll()

        # If we have the UID, continue
        if status == MIFAREReader.MI_OK:

            # Print UID
            # print ("Card read UID: "+str(uid[0])+","+str(uid[1])+","+str(uid[2])+","+str(uid[3])+','+str(uid[4]))
            # This is the default key for authentication
            key = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]

            return 1

            # Select the scanned tag


def read_rfid():
    # if continue_reading:
    while continue_reading:

        # Scan for cards
        (status, TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

        # Get the UID of the card
        (status, uid) = MIFAREReader.MFRC522_Anticoll()

        # If we have the UID, continue
        if status == MIFAREReader.MI_OK:

            # Print UID
            # print ("Card read UID: "+str(uid[0])+","+str(uid[1])+","+str(uid[2])+","+str(uid[3])+','+str(uid[4]))
            # This is the default key for authentication
            key = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
            GPIO.cleanup()
            return uid

            # Select the scanned tag
            MIFAREReader.MFRC522_SelectTag(uid)


def rfid(uid):
    uid1 = [184, 199, 84, 18, 57]
    uid2 = [184, 190, 55, 18, 35]
    uid3 = [89, 84, 49, 24, 36]
    uid4 = [98, 220, 177, 81, 94]
    uid5 = [98, 138, 208, 81, 105]
    uid6 = [114, 87, 131, 81, 247]

    # Check to see if card UID read matches your card UID
    if uid == uid1:
        return "Eom Jeehye"

    if uid == uid2:
        return "Jung Geonggeun"

    if uid == uid3:
        return "Lee Jihyeon"

    if uid == uid4:
        return "Park Kisang"

    if uid == uid5:
        return "SSal Saewookkang"

    if uid == uid6:
        return "Maewoon Saewookkang"


"""uid = read_rfid()
text = rfid(uid)
print(text)"""
