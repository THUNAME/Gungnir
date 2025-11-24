input_file = "/mnt/data/Gungnir/data/Population_Gungnir.csv"
output_file = "/mnt/data/Gungnir/data/Population_Gungnir1.csv"

header = "as,org_name,category,sub_category,routing_prefix,prefix,active_type"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    first = True
    for line in fin:
        line_strip = line.strip()
        # 第一行直接写入
        if first:
            fout.write(line)
            first = False
            continue
        
        # 不是第一行时：检测是否为重复表头，若不是就输出
        if line_strip == header:
            continue
        fout.write(line)
