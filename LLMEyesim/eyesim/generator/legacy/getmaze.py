import numpy as np

# 定义数组（仅用于确定迷宫的大小，这里内容不影响迷宫生成）
array = np.array([
[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
])

def generate_blank_maze(array, scaling=360):
    """
    根据给定的二维数组生成空白迷宫的ASCII表示，并确保顶部和底部边界对齐。

    参数:
    - array (np.ndarray): 二维数组，用于确定迷宫的大小（行数和列数）。
    - scaling (int): 缩放因子，默认为360。

    返回:
    - maze_lines (list of str): 迷宫的每一行字符串列表。
    """
    rows, cols = array.shape
    maze_lines = []

    # 生成顶部边界，并确保长度一致
    top_boundary = " " + "_ " * cols 
    maze_lines.append(top_boundary)

    # 生成中间的行（仅左右边界，无内部墙壁）
    for _ in range(rows-1):
        line = "|" + " " * (cols * 2 - 1) + "|"
        maze_lines.append(line)

    # 生成底部边界
    bottom_boundary = "|" + "_ " * (cols - 1) + "_" + "|"
    maze_lines.append(bottom_boundary)

    # 添加缩放因子
    maze_lines.append(str(scaling))

    return maze_lines

def save_maze_to_file(maze_lines, filename):
    """
    将迷宫的ASCII表示保存到文件中。

    参数:
    - maze_lines (list of str): 迷宫的每一行字符串列表。
    - filename (str): 输出文件的名称。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for line in maze_lines:
                file.write(line + "\n")
        print(f"迷宫已成功保存到 {filename}")
    except IOError as e:
        print(f"保存迷宫时出错: {e}")

def generate_obstacle_positions(array, x_scale=360, y_scale=360, x_offset=180, y_offset=180, z=90):
    """
    根据给定的二维数组生成障碍物的位置。

    参数:
    - array (np.ndarray): 二维数组，1表示障碍物，0表示空位。
    - x_scale (int): x轴的缩放因子。
    - y_scale (int): y轴的缩放因子。
    - x_offset (int): x轴的偏移量。
    - y_offset (int): y轴的偏移量。
    - z (int): z轴的固定值。

    返回:
    - positions (list of tuples): 障碍物的位置列表，每个位置为(x, y, z)。
    """
    positions = []
    rows, cols = array.shape
    for row in range(rows):
        for col in range(cols):
            if array[row, col] == 1:
                x = x_scale * col + x_offset
                # 竖直反转 y 坐标
                y = y_scale * (rows - 1 - row) + y_offset
                positions.append((x, y, z))
    return positions

def generate_sim_file(maze_filename, obstacle_positions, filename):
    """
    将迷宫文件和障碍物位置保存到一个sim文件中。

    参数:
    - maze_filename (str): 迷宫文件的名称，例如 'maze_8x8.maz'。
    - obstacle_positions (list of tuples): 障碍物的位置列表，每个位置为(x, y, z)。
    - filename (str): 输出文件的名称。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            # 写入迷宫部分
            file.write(f'world "{maze_filename}"\n\n')

            # 写入障碍物部分
            for pos in obstacle_positions:
                line = f"Crate1 {pos[0]} {pos[1]} {pos[2]}"
                file.write(line + "\n")
        print(f"Sim文件已成功保存到 {filename}")
    except IOError as e:
        print(f"保存Sim文件时出错: {e}")

def main():
    # 获取数组的尺寸
    rows, cols = array.shape

    # 根据数组尺寸生成文件名
    maze_filename = f'maze_{rows}x{cols}.maz'
    sim_filename = f'maze_sim_{rows}x{cols}.sim'

    # 生成空白迷宫
    maze = generate_blank_maze(array)

    # 打印迷宫到控制台（可选）
    print("生成的空白迷宫：")
    for line in maze:
        print(line)

    # 保存迷宫到文件
    save_maze_to_file(maze, maze_filename)

    # 生成障碍物的位置
    obstacle_positions = generate_obstacle_positions(array)

    # 打印障碍物位置到控制台（可选）
    print("\n生成的障碍物位置列表 (xxs x y z)：")
    for pos in obstacle_positions:
        print(f"xxs {pos[0]} {pos[1]} {pos[2]}")

    # 生成并保存Sim文件
    generate_sim_file(maze_filename, obstacle_positions, sim_filename)

if __name__ == "__main__":
    main()
