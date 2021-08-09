def get_num(row, col):
    return row * 10 + col

width = 9
height = 4
for row in range(0, height):
    for col in range(row + 1, width - row + 1):
        room_num = get_num(row, col)
        with open(f'room-{room_num}.json', 'x') as f:
            f.write('{\n')
            f.write(f'"data_file": "room-{room_num}.png"')
            if col > row + 1:
                f.write(',\n')
                f.write(f'"left_neighbour": "room-{get_num(row, col - 1)}"')
            if col < width - row:
                f.write(',\n')
                f.write(f'"right_neighbour": "room-{get_num(row, col + 1)}"')
            if row > 0:
                f.write(',\n')
                f.write(f'"bottom_neighbour": "room-{get_num(row - 1, col)}"')
            if row < 3:
                f.write(',\n')
                f.write(f'"top_neighbour": "room-{get_num(row + 1, col)}"')
            f.write('\n}\n')
