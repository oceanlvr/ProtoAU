def process_data(file_path):
    user_set = set()
    item_set = set()
    total_interactions = 0

    # 读取和解析文件
    with open(file_path, 'r') as file:
        for line in file:
            user_id, item_id, interactions = line.strip().split()
            user_set.add(user_id)
            item_set.add(item_id)
            total_interactions += int(interactions)

    # 计算用户数和物品数
    num_users = len(user_set)
    num_items = len(item_set)

    # 计算交互密度
    interaction_density = total_interactions / (num_users * num_items)

    return num_users, num_items,total_interactions, interaction_density

# 假设数据文件名为 'interaction_data.txt'
file_path = input("data? ")

num_users, num_items,total_interactions, interaction_density = process_data(file_path)

print(f"用户数: {num_users}")
print(f"物品数: {num_items}")
print(f"交互数: {total_interactions}")
print(f"交互密度: {interaction_density}")
