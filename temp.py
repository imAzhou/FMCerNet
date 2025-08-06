import pandas as pd
from prettytable import PrettyTable


dataconfig = {
    'puretrain': 'data_resource/0630/4_pure_train.csv',
    'jfswtrain': 'data_resource/0630/5_jfsw_train.csv',
    'val': 'data_resource/0630/6_val.csv',
    'test': 'data_resource/0630/7_test.csv',
}

for mode, filepath in dataconfig.items():
    df_data = pd.read_csv(filepath)
    df_unique = df_data.drop_duplicates(subset=["patientId"])
    
    # 统计每个类别的数量
    cls_counts = df_unique["kfb_clsname"].value_counts().to_dict()
    
    # 打印结果
    table = PrettyTable()
    table.title = f"{mode} Class Distribution"
    table.field_names = ["Class Name", "Patient Count"]
    
    total_pos, total_neg = 0, 0
    for cls, count in cls_counts.items():
        table.add_row([cls, count])
        if cls != "NILM":
            total_pos += count
        else:
            total_neg += count
    
    # 加总计行
    table.add_row(["---", "---"])
    table.add_row(["Total", f"Pos:{total_pos} / Neg:{total_neg}"])

    print(table)
