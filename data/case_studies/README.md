# Case Studies Input Data

This directory contains input JSON files for the 5 case studies used in the paper:

1. **AD_input.json** - Autonomous Driving system
2. **ATM_input.json** - Automated Teller Machine system  
3. **Library_input.json** - Library Management system
4. **RollCall_input.json** - Roll Call system
5. **Bookkeeping_input.json** - Bookkeeping system

## Format

Each input file must use the shared OpenRE-Bench case shape:

- `case_name`: stable case identifier used in run ids and reports
- `case_description`: short description of the case domain
- `requirement`: source requirement text used as the benchmark input

MARE, iReDev, and QUARE all consume this same shape.
