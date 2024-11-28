# TODO

## FEATURES

- Out of Stock Flag:
  - from date of analysis to 120 days later (4 months)
    - this could be a variable in general
  - mark which SKUs will be out-of-stock 120 days later

- should read PO data directly from PO data file
  - update: will be prepared separately/specially for consumption by this program

- have analysis input happen through an Excel file that the analyst controls?
  - or just print out what all the options are, and maybe have some checking rules to warn analyst if some settings have values which might not be "optimal"
  - alternatively it can be a Shiny for Python UI?

- option to output analysis results in a format that is easy to read for analyst
  - will require learning how to format using fastexcel writers
  - similar to the format currently used in the main program file?
    - rows per SKU,
      - columns per month
        - show monthly ratios, week fractions, estimate source (PO, or E, or if past, actual sales)
      - columns for:
        - warehouse stock
        - channel stock
        - total estimate before full box logic / splitting
        - total estimate after full box logic / splitting
        - flags column

- use a logging library to log important warnings and errors

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
