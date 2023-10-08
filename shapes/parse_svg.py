import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def extract_coords_from_svg(svg_file_path):
    # Parse the SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    paths = root.findall(".//svg:path", {'svg': 'http://www.w3.org/2000/svg'})

    # Store coordinates
    coordinates = []
    for path in paths:
        d = path.get('d', "")
        commands = d.split()

        for cmd in commands:
            if ',' in cmd:
                x, y = cmd.split(',')
                try:
                    # Scale coordinates
                    x = round(float(x)/100, 6)
                    y = round(float(y)/100, 6)
                    coordinates.append((x, y))
                except:
                    pass

    return coordinates

svg_file_path = './shapes/modular.svg'
coords = extract_coords_from_svg(svg_file_path)
points = np.array(list(set(coords)))


# Plotting using matplotlib
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1])
plt.gca().invert_yaxis()
plt.title("Unique SVG Path Coordinates")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()


# Save to csv
np.savetxt("./shapes/modular.csv", points, delimiter=",", fmt='%f')


# load csv
print(np.loadtxt("./shapes/modular.csv", delimiter=","))