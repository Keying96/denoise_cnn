#!/usr/bin/env python
# encoding: utf-8

import  os
import xlsxwriter

output_dir = "./dataset/Sony/output_20200304/"
pnsr_path = os.path.join(output_dir, "pnsr_{}.xlsx".format(20200304))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# if not os.path.isdir(pnsr_path):
#     os.makefile(pnsr_path)


workbook = xlsxwriter.Workbook(pnsr_path)
worksheet = workbook.add_worksheet()

psnr_data = {}
# psnr_data.update(["zhu" : 487837887])
# psnr_data.update(["hell": 344])
# psnr_data.update(["fd" ,5757])
# psnr_data.update(("zhu" , 487837887))
# psnr_data.update(("hell", 344))
# psnr_data.update(("fd" ,5757))
psnr_data.update({"zhu" : 487837887})
psnr_data.update({"hell" : 344})
psnr_data.update({"fd" : 5757})

row = 0
col = 0
for key, value in (psnr_data.items()):
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, value)
    row += 1

workbook.close()

# csv_file = open(pnsr_path, "w")
# writer = csv.writer(csv_file, lineterminator = '\n')
#
# data_psnr = {"A" : '2444',
#              "B" : "56565",
#              "C": "45454"}
# avg_psnr = 0.0
#
# writer.writerows(data_psnr)
# csv_file.close()



