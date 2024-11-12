import numpy as np
import matplotlib.pyplot as plt


def parse_sdt_file(file_path):
    data = []
    current_stroke = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split()
            if parts[0] == '-1':
                if current_stroke:
                    data.append(current_stroke)
                    current_stroke = []
            else:
                x, y, pressure, direction, altitude, time = map(int, parts)
                current_stroke.append((x, y, pressure, direction, altitude, time))
    if current_stroke:
        data.append(current_stroke)
    return data


def calculate_stroke_distance(stroke1, stroke2):
    total_distance = 0
    n = min(len(stroke1), len(stroke2))
    for i in range(n):
        x1, y1 = stroke1[i][:2]
        x2, y2 = stroke2[i][:2]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total_distance += distance
    average_distance = total_distance / n
    return average_distance


def calculate_character_distance(signature1, signature2):
    total_distance = 0
    N = min(len(signature1), len(signature2))
    for i in range(N):
        stroke_distance = calculate_stroke_distance(signature1[i], signature2[i])
        total_distance += stroke_distance
    return total_distance


def dp_matching(stroke1, stroke2):
    n = len(stroke1)
    m = len(stroke2)
    dp = np.full((n + 1, m + 1), float('inf'))
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt((stroke1[i - 1][0] - stroke2[j - 1][0]) ** 2 + (stroke1[i - 1][1] - stroke2[j - 1][1]) ** 2)
            dp[i, j] = min(dp[i - 1, j - 1] + cost, dp[i - 1, j] + cost, dp[i, j - 1] + cost)
    return dp[n, m]


def calculate_dp_character_distance(signature1, signature2):
    total_distance = 0
    N = min(len(signature1), len(signature2))
    for i in range(N):
        stroke_distance = dp_matching(signature1[i], signature2[i])
        total_distance += stroke_distance
    return total_distance


def plot_matching(signature1, signature2):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    scale = 1.0

    # Plot signatures
    for stroke1, stroke2 in zip(signature1, signature2):
        x1, y1 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke1])
        x2, y2 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke2])
        axs[0].plot(x1, y1, 'r-')
        axs[0].plot(x2, y2, 'b-')

    axs[0].set_title('Signatures')
    axs[0].legend(['Signature 1', 'Signature 2'])

    # Plot linear matching
    for stroke1, stroke2 in zip(signature1, signature2):
        x1, y1 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke1])
        x2, y2 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke2])
        for (x1i, y1i), (x2i, y2i) in zip(zip(x1, y1), zip(x2, y2)):
            axs[1].plot([x1i, x2i], [y1i, y2i], 'grey', linestyle='--', linewidth=0.3)

    axs[1].set_title('Linear Matching')
    axs[1].legend(['Matching'])

    # Plot DP matching
    for stroke1, stroke2 in zip(signature1, signature2):
        x1, y1 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke1])
        x2, y2 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke2])
        for (x1i, y1i), (x2i, y2i) in zip(zip(x1, y1), zip(x2, y2)):
            axs[2].plot([x1i, x2i], [y1i + 20, y2i + 20], 'green', linestyle='--', linewidth=0.3)

    axs[2].set_title('DP Matching')
    axs[2].legend(['Matching'])

    plt.show()


def main():
    signature_data1 = parse_sdt_file('Otherfiles/code/database/ref.sdt')
    signature_data2 = parse_sdt_file('Otherfiles/code/database/ref.sdt')

    character_distance_linear = calculate_character_distance(signature_data1, signature_data2)
    print("Linear Matching Calculated Distance between characters:", character_distance_linear)

    character_distance_dp = calculate_dp_character_distance(signature_data1, signature_data2)
    print("DP Matching Calculated Distance between characters:", character_distance_dp)

    plot_matching(signature_data1, signature_data2)


if __name__ == '__main__':
    main()
