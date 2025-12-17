import pandas as pd
import os

def main():
    # === 1. 读取数据 ===
    df_train = pd.read_csv('data_resource/0630/45_0924_train.csv')
    df_val = pd.read_csv('data_resource/0630/67_0924_val.csv')
    df_data = pd.concat([df_train, df_val], ignore_index=True)

    df_error = pd.read_csv('cj_test_temporary/1_result.csv')

    # === 2. 统计并筛选加和值 ≤ 1 的行 ===
    label_cols = ['cell_detector_label', 'ml_decoder_label', 'ours_WS800_label']
    df_error['label_sum'] = df_error[label_cols].sum(axis=1)
    df_filtered = df_error[df_error['label_sum'] <= 1].copy()

    print(f"筛选后共有 {len(df_filtered)} 条样本满足条件。")

    # === 3. 通过 file_path 匹配 df_data 中的 kfb_path ===
    # 检查是否存在重复 kfb_path
    dup_paths = df_data[df_data.duplicated(subset='kfb_path', keep=False)]
    if not dup_paths.empty:
        print(f"⚠️ 检测到 {len(dup_paths)} 条重复的 kfb_path，去重处理。")
        df_data = df_data.drop_duplicates(subset='kfb_path', keep='first')

    # 构建映射字典
    source_map = df_data.set_index('kfb_path')[['patientId', 'kfb_clsname']].to_dict('index')

    matched_rows = []
    for _, row in df_filtered.iterrows():
        file_path = row['file_path']
        if file_path in source_map:
            info = source_map[file_path]
            matched_rows.append({
                'file_path': file_path,
                'patientId': info['patientId'],
                'kfb_clsname': info['kfb_clsname']
            })

    df_matched = pd.DataFrame(matched_rows)

    # === 4. 保存结果 ===
    save_dir = 'statistic_results/1103'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'filtered_mismatched_cases.csv')
    df_matched.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存结果到: {save_path}")

    # === 5. 统计每个 kfb_clsname 的数量 ===
    cls_count = df_matched['kfb_clsname'].value_counts().reset_index()
    cls_count.columns = ['kfb_clsname', 'count']
    print("\n每个类别对应的行数：")
    print(cls_count)

    count_path = os.path.join(save_dir, 'filtered_class_count.csv')
    cls_count.to_csv(count_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存类别统计到: {count_path}")

def collect_error2csv():
    input_csv = 'statistic_results/1103/filtered_mismatched_cases.csv'
    save_path = 'statistic_results/1103/mismatched_upload.csv'
    sample_class='ASC-US'
    sample_num=30
    random_state=42

    # === 1. 读取数据 ===
    df = pd.read_csv(input_csv)
    print(f"读取 {len(df)} 条记录。")

    # === 2. 按类别划分 ===
    df_target = df[df['kfb_clsname'] == sample_class]
    df_others = df[df['kfb_clsname'] != sample_class]

    print(f"{sample_class} 类别共有 {len(df_target)} 条样本。")

    # === 3. 随机抽样 ===
    if len(df_target) > sample_num:
        df_target_sampled = df_target.sample(n=sample_num, random_state=random_state)
        print(f"已随机抽取 {sample_num} 条 {sample_class} 样本。")
    else:
        df_target_sampled = df_target
        print(f"⚠️ {sample_class} 类别不足 {sample_num} 条，保留全部 {len(df_target)} 条。")

    # === 4. 合并数据 ===
    df_final = pd.concat([df_others, df_target_sampled], ignore_index=True)

    # === 5. 保存结果 ===
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存筛选结果到: {save_path}")
    print(f"最终共 {len(df_final)} 条样本。")

    # === 6. 打印类别分布 ===
    print("\n最终类别分布：")
    print(df_final['kfb_clsname'].value_counts())

if __name__ == '__main__':
    # main()

    collect_error2csv()