# OpenRE-Bench Mining Statistics

Per-project issue funnel from raw issues to the final aggregated case. Each column after `raw` shows survivors after that filter stage.

> `Raw` is the fetched issue sample (most-recent label-matching issues, capped at `github.max_issues_per_project`), not the project's all-time issue total. The `Label` drop is typically 0 because the API query already restricts to the configured labels.

| Project | Domain | Raw | Labels | State | Length | Bug | Signal | Discussion | Dedup | Final Issues | Words | Window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch/pytorch | Deep Learning | 600 | 600 | 211 | 206 | 132 | 132 | 132 | 131 | 4 | 380 | 2026q1 |
| apache/airflow | Workflow Orchestration | 600 | 600 | 429 | 406 | 367 | 367 | 367 | 358 | 4 | 387 | 2026q2 |
| python-poetry/poetry | Packaging | 600 | 600 | 502 | 484 | 419 | 419 | 419 | 419 | 3 | 363 | 2022q1 |
| matplotlib/matplotlib | Visualization | 600 | 600 | 409 | 386 | 356 | 356 | 356 | 355 | 4 | 390 | 2021q1 |
| streamlit/streamlit | Web App Framework | 600 | 600 | 326 | 325 | 271 | 271 | 271 | 269 | 4 | 426 | 2025q3 |
| microsoft/TypeScript | Programming Language | 600 | 600 | 242 | 235 | 173 | 173 | 173 | 173 | 3 | 363 | 2024q1 |
| angular/angular | Frontend Framework | 600 | 600 | 547 | 542 | 427 | 427 | 425 | 425 | 3 | 363 | 2020q4 |
| grafana/grafana | Observability | 600 | 600 | 283 | 250 | 218 | 218 | 218 | 215 | 3 | 310 | 2025q3 |
| react/react | UI Library | 350 | 350 | 324 | 303 | 168 | 168 | 156 | 156 | 4 | 416 | 2016q2 |
| nodejs/node | JS Runtime | 600 | 600 | 518 | 500 | 420 | 420 | 419 | 417 | 4 | 385 | 2025q1 |
| spring-projects/spring-boot | Application Framework | 600 | 600 | 502 | 270 | 251 | 251 | 251 | 249 | 4 | 403 | 2025q3 |
| keycloak/keycloak | Identity / Security | 600 | 600 | 324 | 308 | 281 | 281 | 278 | 278 | 3 | 356 | 2024q1 |
| kubernetes/kubernetes | Container Orchestration | 600 | 600 | 382 | 349 | 307 | 307 | 305 | 305 | 4 | 341 | 2025q1 |
| hashicorp/terraform | Infrastructure as Code | 600 | 600 | 385 | 379 | 322 | 322 | 322 | 322 | 4 | 393 | 2023q4 |
| rust-lang/rust | Compiler | 600 | 600 | 321 | 270 | 203 | 203 | 200 | 200 | 3 | 323 | 2020q3 |
| tauri-apps/tauri | Desktop App Framework | 600 | 600 | 383 | 373 | 329 | 329 | 328 | 328 | 3 | 363 | 2024q4 |

## Issues Dropped by Each Filter

| Project | Label | State | Length | Bug | Signal | Discussion | Dedup |
|---|---|---|---|---|---|---|---|
| pytorch/pytorch | 0 | 389 | 5 | 74 | 0 | 0 | 1 |
| apache/airflow | 0 | 171 | 23 | 39 | 0 | 0 | 9 |
| python-poetry/poetry | 0 | 98 | 18 | 65 | 0 | 0 | 0 |
| matplotlib/matplotlib | 0 | 191 | 23 | 30 | 0 | 0 | 1 |
| streamlit/streamlit | 0 | 274 | 1 | 54 | 0 | 0 | 2 |
| microsoft/TypeScript | 0 | 358 | 7 | 62 | 0 | 0 | 0 |
| angular/angular | 0 | 53 | 5 | 115 | 0 | 2 | 0 |
| grafana/grafana | 0 | 317 | 33 | 32 | 0 | 0 | 3 |
| react/react | 0 | 26 | 21 | 135 | 0 | 12 | 0 |
| nodejs/node | 0 | 82 | 18 | 80 | 0 | 1 | 2 |
| spring-projects/spring-boot | 0 | 98 | 232 | 19 | 0 | 0 | 2 |
| keycloak/keycloak | 0 | 276 | 16 | 27 | 0 | 3 | 0 |
| kubernetes/kubernetes | 0 | 218 | 33 | 42 | 0 | 2 | 0 |
| hashicorp/terraform | 0 | 215 | 6 | 57 | 0 | 0 | 0 |
| rust-lang/rust | 0 | 279 | 51 | 67 | 0 | 3 | 0 |
| tauri-apps/tauri | 0 | 217 | 10 | 44 | 0 | 1 | 0 |

**Totals:** 9350 raw issues across 16 projects -> 57 issues in 16 final case(s).
