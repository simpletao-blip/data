# KAUST 2023 LBV feasibility record

- Date: 2026-07-05 (Asia/Shanghai)
- Mechanism: `mechanisms/converted/KAUST_2023.yaml`
- Design case: first row of `data/processed/lbv_validation_design.csv`
- Solver: staged mixture-averaged -> multicomponent -> Soret `FreeFlame`
- Wall-time limit: 900 s
- Outcome: no completed solution or result row before the external 900.3 s limit
- Observed peak working set: approximately 0.58 GB for the solver process
- Admission decision: retain for 0D IDT and JSR validation; exclude from the
  formal 80-case LBV ensemble on computational-feasibility grounds.

This is not classified as a kinetic-model prediction failure. The uncompleted
case is not included in error statistics, and no missing value is imputed.
