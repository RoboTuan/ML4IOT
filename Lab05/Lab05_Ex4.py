import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('command', nargs=1, type=str)
parser.add_argument('operands', nargs='+', type=str)
args = parser.parse_args()

command = args.command[0]
operands = args.operands

url = 'http://0.0.0.0:8080/{}?'.format(command)

for i, op in enumerate(operands):
    url += '&op{}={}'.format(i+1, op)

print(url)

r = requests.get(url)

if r.status_code == 200:
    # body will be a dictionary
    body = r.json()
    commands = {"add":"+", "sub":"-", "mul":"*", "div":"/"}

    message = operands[0]

    for i in range(1, len(operands)):
        message += '{}{}'.format(commands[body["command"]], operands[i])

    print(message, "=", body["result"])
else:
    print("Error: ", r.status_code)
    print(r.text)