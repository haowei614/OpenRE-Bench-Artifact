# MARE vs iReDev vs QUARE: Three-Way Experimental Comparison (Updated)

**Date**: 2026-02-26  
**Total Runs**: 180 (60 MARE + 60 iReDev + 60 QUARE)  
**Model**: gpt-4o-mini-2024-07-18 | **Temperature**: 0.7 | **Round Cap**: 3 | **Seeds**: 101, 202, 303  
**Cases**: AD, ATM, Library, RollCall, Bookkeeping

> [!NOTE]
> Validity snapshot from latest metrics summary: MARE **60/0**, iReDev **15/45**, QUARE **0/60** (valid/invalid runs).

## 1. Requirement Volume (NIV)

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 23.0 | 29.0 | **35.0** |
| ATM | 25.0 | 28.0 | **35.0** |
| Library | 24.3 | 28.0 | **35.0** |
| RollCall | 25.0 | 27.3 | **35.0** |
| Bookkeeping | 24.7 | 28.0 | **35.0** |
| **Average** | **24.4** | **28.1** | **35.0** |

## 2. CHV / MDC (NIV Average)

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| CHV | 0.00476 | 0.00636 | 0.004309 |
| MDC | 0.835 | 0.705 | 0.672927 |

## 3. Semantic Preservation / CRR (NIV Average)

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| BERTScore P3 vs P1 | 89.0% | 92.6% | 94.8% |
| BERTScore P2 vs P1 | 97.0% | 99.4% | 100.0% |
| Conflict Resolution Rate | 66.7% | 46.7% | 24.9% |

## 4. Compliance / Runtime (NIV Average)

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| Compliance Coverage | 47.6% | 47.8% | 98.3% |
| Runtime | 38.4s | 166.5s | 48.6s |

## 5. Runtime by Setting

| Setting | MARE (s) | iReDev (s) | QUARE (s) |
|---|---:|---:|---:|
| single_agent | 0.005 | 0.006 | 0.003 |
| multi_agent_without_negotiation | 39.634 | 189.715 | 0.007 |
| multi_agent_with_negotiation | 35.210 | 179.106 | 59.983 |
| negotiation_integration_verification | 38.429 | 166.488 | 48.554 |

*This report is regenerated from latest local artifacts and reflects the updated QUARE generation behavior (non-fixed per-axis counts).*
