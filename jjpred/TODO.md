# TODO

## FEATURES

- Out of Stock Flag:
  - from date of analysis to 120 days later (4 months)
    - this could be a variable in general
  - mark which SKUs will be out-of-stock 120 days later

- have analysis input happen through an Excel file that the analyst controls?
  - or just print out what all the options are, and maybe have some checking rules to warn analyst if some settings have values which might not be "optimal"
  - alternatively it can be a Shiny for Python UI!

- analysis output should be output to Excel in a format that is easy to read for analyst
  - will require learning how to format using fastexcel...
  - similar to the format currently used in the main program file?

- use a logging library to log important warnings and errors

- should read QTY/BOX info from Excel file in `TWK Matt > PO` folder
  - PO boxes and volume - All seasons.xlsx
  -
- should read PO data directly from PO data file
  - in `TWK Matt > PO` folder?
    - 2{n}{season} PO Plan All CATs - {YYYY.MM.DD} - v{m}.xlsx
      - e.g. for 2024 SS: 24SS PO Plan All CATs - 2023.07.01 - v3.xlsx
    - read "Plan Vancouver" sheet for NA data
      - and "Plan UK/DE" for EU data?
  - store copied files in ExtractedPOData folder

## NOTES

- confirm/fix logic for how PO data is read from season file
  - currently we are doing:
    - if dispatch season is X (e.g. X=SS) and Y is other season (e.g. Y=FW):
      - then check to see the X sheet for items
      - for those item not in the X sheet, check the Y sheet
    - but there is some uncertainty on whether this is sufficient

- for XPC use combination of XBK and XBM as reference category

- for IPS use reference category WMT (big sales category) or IHT (small sales category)
  - all I* category can use WMT or IHT as reference category

- the AJA series: should use WJT as the reference category
  - also should have marketing config of 5 units
