import json 

def increase_count():
    try:
        with open('usage_count.json', 'r') as file:
            data = json.load(file)
            count = data.get('count', 0)
    except FileNotFoundError:
        count = 0
    
    count += 1
    data = {'count': count}
    with open('usage_count.json', 'w') as file:
        json.dump(data, file)
