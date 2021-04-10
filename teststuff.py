import functionsData as fd
import pandas as pd
descr_ciks = ['0000072971', '0001403161', '0000875320', '0001318605', \
              '0000078003', '0001021860', '0000879101', '0000019617', \
              '0000886982', '0000037996', '0000034088', '0000712515', \
              '0000732717', '0000320193', '0000789019', '0000106640', \
              '0001418091', '0001283699', '0000092380', '0001039684']
drive = '/Volumes/LaCie/Data/'
fullCIKs = fd.loadFile(drive+'CIKs_final.pckl')
desc = fd.ciksDescriptives(descr_ciks)

name_xlsx = drive+'descriptivesCIK.xlsx'
writer = pd.ExcelWriter(name_xlsx,engine='xlsxwriter')
workbook=writer.book
for key in desc.keys():
    worksheet=workbook.add_worksheet(key)
    writer.sheets[key] = worksheet
    
    worksheet.write_string(0, 0, 'General Descriptives')
    desc[key]['Descriptives'].to_excel(writer,sheet_name=key,startrow=1 , startcol=0)
    
    worksheet.write_string(desc[key]['Descriptives'].shape[0] + 4, 0, 'Quantiles')
    desc[key]['Quantiles'].to_excel(writer,sheet_name=key,startrow=desc[key]['Descriptives'].shape[0] + 5, startcol=0)
    
    worksheet.write_string(desc[key]['Descriptives'].shape[0] + 5 + desc[key]['Quantiles'].shape[0] + 4, 0, 'Periods')
    desc[key]['Periods'].to_excel(writer,sheet_name=key,startrow=desc[key]['Descriptives'].shape[0] + 5 + desc[key]['Quantiles'].shape[0] + 5, startcol=0)
writer.save()