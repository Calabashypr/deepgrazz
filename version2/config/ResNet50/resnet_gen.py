import json

# with open('network.json', 'w', encoding='utf8') as f:
#     network = {"network": []}
#     for i in range(130):
#         node = {"id": i, "name": "", "params": {}, "state": "none", "to": [i + 1], "from": [i - 1]}
#         if i == 0:
#             node["state"] = "src"
#         network["network"].append(node)
#     json.dump(network, f)


with open('network.json', 'r', encoding='utf8') as f:
    network = json.load(f)
    print(network)
    nodes = network['network']
    for node in nodes:
        if node['name'] == 'Conv2D':
            node['params'] = {'in_channels': '', 'out_channels': '', 'kernel_size': ''}

print(network)

with open('network_params.json', 'w', encoding='utf8') as f:
    json.dump(network, f)
