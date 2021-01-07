import json
from datetime import datetime
import sys
import threading
import time


class Catalog():

    def __init__(self, remove_old = False):
        self.remove_old_devices_feature = remove_old
    

    # end_points= {topics:[..., ...], rest_web_services: [..., ...]}
    def add_device(self, end_points, resources):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)

        now = datetime.now()
        timestamp = int(now.timestamp())

        try:
            id = int(data["devices"][-1]["id"]) + 1
        except IndexError:
            print("First device added")
            id =  0

        device = {
            "id" : id,
            "insert-timestamp": timestamp,
            "end_point": end_points,
            "resources": resources}

        data["devices"].append(device)

        with open('./Lab06/devices_users.json', 'w') as outfile:
            json.dump(data, outfile)

    
    def get_devices(self):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)
        return  data["devices"]

    
    def get_device_id(self, id):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)
        
        for device in data["devices"]:
            if device["id"] == id:
                return device
        
        # Manage in the receiver with try/except
        print("No devices with such id")
        return None

    
    def register_user(self, name, surname):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)        

        try:
            id = int(data["users"][-1]["id"]) + 1
        except IndexError:
            print("First user added")
            id =  0

        user = {
            "id": id,
            "name": name,
            "surname": surname
        }

        data["users"].append(user)

        with open('./Lab06/devices_users.json', 'w') as outfile:
            json.dump(data, outfile)


    def get_users(self):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)
        return  data["users"]

    
    def get_user_id(self, id):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)
        
        for user in data["users"]:
            if user["id"] == id:
                return user
        
        # Manage in the receiver with try/except
        print("No user with such id")
        return None

    
    def remove_old_devices(self):
        with open('./Lab06/devices_users.json') as json_file:
            data = json.load(json_file)

            while True:
                now = datetime.now()
                timestamp = int(now.timestamp())

                devices = data["devices"]

                for device in devices:
                    if int(timestamp) - device["insert-timestamp"] > 20:
                        devices.remove(device)
                        print(f"Removed device:\n{device}")

                data["devices"] = devices
                
                with open('./Lab06/devices_users.json', 'w') as outfile:
                    json.dump(data, outfile)

                time.sleep(20)

    
    def run(self):

        if self.remove_old_devices_feature is True:
            t1 = threading.Thread(target=self.remove_old_devices)
            t1.start()



superCatalog = Catalog()

# superCatalog.add_device({"topics":["/276033/temperature", "/276033/microphone"]}, ["humidity", "temperature", "microphone"])

# superCatalog.register_user("Paolo", "Bianchi")

# print(superCatalog.get_devices())
# print(superCatalog.get_users())
# print(superCatalog.get_device_id(11))
# print(superCatalog.get_device_id(1))
# print(superCatalog.get_user_id(0))
# print(superCatalog.get_user_id(9))

superCatalog.run()

while True:
    print("Devices")
    print(superCatalog.get_devices())
    superCatalog.add_device({"topics":["/276033/temperature",]}, ["temperature",])
    time.sleep(10)

