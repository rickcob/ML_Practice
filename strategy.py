# RANDOM SEARCH
# def strategy(history):
#     x,y = np.random.randint(0, const.SIZE), np.random.randint(0, const.SIZE)
#     return (x, y)


# BRUTE FORCE SEARCH
# def strategy(history):
#     a = math.ceil(np.sqrt(const.TURNS))
#     stride = 200 / a
#     x, y = int(stride * (len(history) // a + 1 / 2)),
#            int(stride * (len(history) % a + 1 / 2))
#     return (x, y)


# def strategy(history):
#     a = math.ceil(np.sqrt(const.TURNS))
#     if len(history) == 0:
#         x, y = np.random.randint(0, const.SIZE-a),
#                np.random.randint(0, const.SIZE-a)
#     else:
#         x_0, y_0 = history[0][0]

#         x, y = int(x_0 + (len(history)) % a), int(y_0 + (len(history)) // a)
#     return (x, y)


def strategy(history):
    own_history = history.copy()
    distances = []
    if len(history) % 2 == 0:
        own_history.append([(0, 0), 0])
        own_history.append([(200, 200), 0])
    else:
        own_history.append([(0, 200), 0])
        own_history.append([(200, 0), 0])
    for i in range(len(own_history) - 1):
        for j in range(i + 1, len(own_history)):
            dx = (own_history[i][0][0] - own_history[j][0][0]) ** 2
            dy = (own_history[i][0][1] - own_history[j][0][1]) ** 2
            distances.append([own_history[i][0], own_history[j][0], dx + dy])
    max_distance = 0
    for i, distance in enumerate(distances):
        if distance[-1] > max_distance:
            max_distance = distance[-1]
            x = (distance[0][0] + distance[1][0]) / 2
            y = (distance[0][1] + distance[1][1]) / 2
    return (x, y)