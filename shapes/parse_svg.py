import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def extract_coords_from_svg(svg_file_path, BASE):
    assert BASE in ["flame", "modular"]
    scaling = {
        "modular": 100,
        "flame": 250
    }

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
                    scale_factor = scaling[BASE]
                    x = round(float(x)/scale_factor, 6)
                    y = round(float(y)/scale_factor, 6)
                    coordinates.append((x, y))
                except:
                    pass

    return coordinates


def plot_points(points):
    # Plotting using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1])
    plt.gca().invert_yaxis()
    plt.title("Unique SVG Path Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # BASE = "modular"
    BASE = "flame"
    
    
    svg_file_path = f'./shapes/{BASE}.svg'
    coords = extract_coords_from_svg(svg_file_path, BASE)
    points = np.array(list(set(coords)))

    plot_points(points)

    # Save to csv
    np.savetxt(f"./shapes/{BASE}.csv", points, delimiter=",", fmt='%f')


    # load csv
    print(np.loadtxt(f"./shapes/{BASE}.csv", delimiter=","))