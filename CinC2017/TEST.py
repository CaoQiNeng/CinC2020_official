import csv
import numpy as np

with open("result.csv", "w") as datacsv:
    # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
    csvwriter = csv.writer(datacsv, dialect=("excel"))
    # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
    csvwriter.writerow(["rd", "best_f1_mean"])

    rd = np.random.randint(0, 466, 5)

    while (10):

        csvwriter.writerow([str(rd), 1])