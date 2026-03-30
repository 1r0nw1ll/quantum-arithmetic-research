# AIID Sample-50 Through QA Failure Algebra

## Setup
- Source: `incidentdatabase.ai` incidents feed snapshot (50 incidents, IDs 1-50)
- Labeling: manual single-label assignment into `F1..F6` from `QA_FAILURE_ALGEBRA.json`
- Severity analysis A: manual severity rubric (Low/Medium/High/Critical)
- Severity analysis B: AIID `CSETv0` Severity taxonomy when available
- AIID severity coverage in sample: **49 / 50** (`44 / 50` scored after excluding `Unclear/unknown`)

## Updated Conclusions
- The F1-F6 mapping is total on this sample (`50/50` incidents assigned exactly one class), but not perfectly clean: `4/50` cases are marked taxonomy-strain.
- Severity calibration changed materially after switching to AIID-native `CSETv0` severity:
  - F5 remains the highest-severity class directionally.
  - The effect size is weaker than the manual rubric suggested.
- Practical interpretation: F1-F6 appears stronger as a failure-mechanism stratification than as a direct predictor of harm magnitude.
- Recommended refinement is additive, not a redesign: keep F1-F6 primitives and add composition labeling (`primary`, optional `secondary`, `composition_form`) plus explicit `strain_witness`.

## 1) Does Every Incident Map Cleanly To Exactly One Class?
- `all_mapped`: **True**
- `all_singleton_labels`: **True**
- `unmapped_ids`: []
- `duplicate_label_ids`: []

## 2) Do Classes Predict Severity?
### 2A) Manual Severity Rubric
| Class | Name | n | Mean Severity (1-4) | High/Critical Rate |
|---|---|---:|---:|---:|
| F1 | Formalization Gap | 8 | 2.00 | 25.00% |
| F2 | Case Explosion | 4 | 2.50 | 25.00% |
| F3 | Rewrite Blocked | 4 | 2.00 | 0.00% |
| F4 | Budget Exhaustion | 2 | 1.50 | 0.00% |
| F5 | Kernel Violation | 16 | 3.12 | 81.25% |
| F6 | Component Isolation | 16 | 2.31 | 31.25% |

### 2B) AIID CSETv0 Severity (covered subset)
| Class | Covered n | Mean Severity (1-5) | Severe/Critical Rate |
|---|---:|---:|---:|
| F1 | 8 | 1.25 | 0.00% |
| F2 | 4 | 1.25 | 0.00% |
| F3 | 4 | 1.00 | 0.00% |
| F4 | 0 | n/a | n/a |
| F5 | 16 | 2.38 | 31.25% |
| F6 | 12 | 1.50 | 0.00% |

## 3) Do Classes Predict Recurrence?
- Recurrence signal in this study uses two proxies:
  - manual tag: `single | platform_repeat | systemic_series`
  - developer-repeat proxy: incident has a developer appearing >1 time in sample

| Class | Platform/Systemic Repeat Rate | Repeat-Developer Rate |
|---|---:|---:|
| F1 | 12.50% | 25.00% |
| F2 | 25.00% | 0.00% |
| F3 | 75.00% | 75.00% |
| F4 | 50.00% | 50.00% |
| F5 | 43.75% | 50.00% |
| F6 | 56.25% | 56.25% |

## 4) Incidents That Strain The Taxonomy
- `taxonomy_strain_count`: **4 / 50** (8.00%)
- `21` [Tougher Turing Test Exposes Chatbots’ Stupidity (migrated to Issue)](https://incidentdatabase.ai/cite/21) -> F4 (Capability shortfall fits budget/capacity exhaustion more than harm failure.)
- `39` [Deepfake Obama Introduction of Deepfakes](https://incidentdatabase.ai/cite/39) -> F1 (Deepfake emergence highlights missing authenticity formalization in media pipelines.)
- `41` [All Image Captions Produced are Violent](https://incidentdatabase.ai/cite/41) -> F1 (Designed-to-fail experiment sits at boundary between demonstration and incident.)
- `42` [Inefficiencies in the United States Resident Matching Program](https://incidentdatabase.ai/cite/42) -> F2 (Matching inefficiency attributed to combinatorial constraint explosion.)

## 5) Composition Metrics
- `composition_rate`: **44 / 50** (88.00%)
- `composition_form_hist`: `{'parallel': 20, 'serial': 21, 'feedback': 3}`
- `Batch A rule`: deterministic round-robin over non-strain incidents in class order `F1,F2,F3,F4,F5,F6` (take first 10).
- `Batch A selected IDs`: `[1, 7, 33, 30, 2, 11, 9, 28, 34, 3]`
- `Batch A coverage in composition labels`: **10 / 10**
- `Batch B rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.
- `Batch B selected IDs`: `[12, 10, 38, 36, 4, 13, 17, 45, 5, 14]`
- `Batch B coverage in composition labels`: **10 / 10**
- `Batch C rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.
- `Batch C selected IDs`: `[35, 6, 15, 44, 8, 16, 20, 18, 22, 19]`
- `Batch C coverage in composition labels`: **10 / 10**
- `Batch D rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.
- `Batch D selected IDs`: `[23, 29, 24, 32, 25, 37, 26, 40, 27, 43]`
- `Batch D coverage in composition labels`: **10 / 10**
- `Batch A+B+C+D exact match`: **True** (non-strain composed outside A+B+C+D: `[]`)
- `composed_non_strain_count`: **40** (composition now extends beyond the strain-only subset).
| Group | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |
|---|---:|---:|---:|
| composed | 38 | 1.76 | 13.16% |
| primitive_only | 6 | 1.33 | 0.00% |

| Composition Form | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |
|---|---:|---:|---:|
| feedback | 3 | 3.33 | 66.67% |
| parallel | 17 | 1.94 | 11.76% |
| serial | 18 | 1.33 | 5.56% |

| Feedback Risk Table (scored composed only) | Severe/Critical | Not Severe/Critical | Total |
|---|---:|---:|---:|
| feedback | 2 | 1 | 3 |
| nonfeedback | 3 | 32 | 35 |
- `n_feedback_scored`: `3`, `n_nonfeedback_scored`: `35`
- `risk_ratio_feedback_vs_nonfeedback`: **7.778**

| Secondary Presence (within composed) | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |
|---|---:|---:|---:|
| secondary_present | 38 | 1.76 | 13.16% |
| secondary_absent | 0 | n/a | n/a |

## Sample Table (50)
| ID | Date | Class | Manual Severity | AIID Severity | Composed | Form | Repeat | Title |
|---:|---|---|---|---|---|---|---|---|
| 1 | 2015-05-19 | F1 | High | Moderate | yes | parallel | single | [Google’s YouTube Kids App Presents Inappropriate Content](https://incidentdatabase.ai/cite/1) |
| 2 | 2018-12-05 | F5 | High | Moderate | yes | parallel | single | [Warehouse robot ruptures can of bear spray and injures workers](https://incidentdatabase.ai/cite/2) |
| 3 | 2018-10-27 | F5 | Critical | Critical | yes | parallel | single | [Crashes with Maneuvering Characteristics Augmentation System (MCAS)](https://incidentdatabase.ai/cite/3) |
| 4 | 2018-03-18 | F5 | Critical | Severe | yes | serial | platform_repeat | [Uber AV Killed Pedestrian in Arizona](https://incidentdatabase.ai/cite/4) |
| 5 | 2015-07-13 | F5 | Critical | Severe | yes | feedback | systemic_series | [Collection of Robotic Surgery Malfunctions](https://incidentdatabase.ai/cite/5) |
| 6 | 2016-03-24 | F5 | High | Minor | yes | parallel | single | [Microsoft's TayBot Allegedly Posts Racist, Sexist, and Anti-Semitic Content to Twitter](https://incidentdatabase.ai/cite/6) |
| 7 | 2017-02-24 | F2 | Medium | Negligible | yes | parallel | single | [Wikipedia Vandalism Prevention Bot Loop](https://incidentdatabase.ai/cite/7) |
| 8 | 2014-08-15 | F5 | High | Negligible | yes | serial | platform_repeat | [Uber Autonomous Cars Running Red Lights](https://incidentdatabase.ai/cite/8) |
| 9 | 2012-02-25 | F1 | Medium | Negligible | yes | parallel | single | [NY City School Teacher Evaluation Algorithm Contested](https://incidentdatabase.ai/cite/9) |
| 10 | 2014-08-14 | F1 | Medium | Negligible | yes | parallel | single | [Kronos Scheduling Algorithm Allegedly Caused Financial Issues for Starbucks Employees](https://incidentdatabase.ai/cite/10) |
| 11 | 2016-05-23 | F6 | High | Unclear/unknown | yes | parallel | single | [Northpointe Risk Models](https://incidentdatabase.ai/cite/11) |
| 12 | 2016-07-21 | F6 | Medium | Unclear/unknown | yes | serial | platform_repeat | [Common Biases of Vector Embeddings](https://incidentdatabase.ai/cite/12) |
| 13 | 2017-02-27 | F6 | Medium | Minor | yes | serial | platform_repeat | [High-Toxicity Assessed on Text Involving Women and Minority Groups](https://incidentdatabase.ai/cite/13) |
| 14 | 2017-10-26 | F6 | Medium | Negligible | yes | serial | platform_repeat | [Biased Sentiment Analysis](https://incidentdatabase.ai/cite/14) |
| 15 | 2008-05-23 | F6 | Medium | Negligible | yes | serial | platform_repeat | [Amazon Censors Gay Books](https://incidentdatabase.ai/cite/15) |
| 16 | 2015-06-03 | F6 | High | Minor | yes | serial | platform_repeat | [Images of Black People Labeled as Gorillas](https://incidentdatabase.ai/cite/16) |
| 17 | 2015-11-03 | F1 | Low | Negligible | yes | serial | platform_repeat | [Inappropriate Gmail Smart Reply Suggestions](https://incidentdatabase.ai/cite/17) |
| 18 | 2015-04-04 | F6 | Medium | Minor | yes | serial | platform_repeat | [Gender Biases of Google Image Search](https://incidentdatabase.ai/cite/18) |
| 19 | 2013-01-23 | F6 | Medium | Unclear/unknown | yes | serial | platform_repeat | [Sexist and Racist Google Adsense Advertisements](https://incidentdatabase.ai/cite/19) |
| 20 | 2016-06-30 | F5 | High | Severe | yes | feedback | systemic_series | [A Collection of Tesla Autopilot-Involved Crashes](https://incidentdatabase.ai/cite/20) |
| 21 | 2016-07-14 | F4 | Low | Unclear/unknown | yes | parallel | single | [Tougher Turing Test Exposes Chatbots’ Stupidity (migrated to Issue)](https://incidentdatabase.ai/cite/21) |
| 22 | 2017-12-06 | F5 | High | Negligible | yes | serial | platform_repeat | [Waze Navigates Motorists into Wildfires](https://incidentdatabase.ai/cite/22) |
| 23 | 2017-11-08 | F5 | Medium | Minor | yes | parallel | single | [Las Vegas Self-Driving Bus Involved in Accident](https://incidentdatabase.ai/cite/23) |
| 24 | 2014-07-15 | F5 | Critical | Severe | yes | parallel | single | [Robot kills worker at German Volkswagen plant](https://incidentdatabase.ai/cite/24) |
| 25 | 2015-06-23 | F5 | Medium | Negligible | yes | serial | platform_repeat | [Google and Delphi Self-Driving Prototypes Allegedly Involved in Near-Miss on San Antonio Road, Palo Alto](https://incidentdatabase.ai/cite/25) |
| 26 | 2017-09-13 | F5 | Medium | Negligible | yes | serial | platform_repeat | [Hackers Break Apple Face ID](https://incidentdatabase.ai/cite/26) |
| 27 | 1983-09-26 | F5 | Critical | Negligible | yes | parallel | single | [Nuclear False Alarm](https://incidentdatabase.ai/cite/27) |
| 28 | 2010-05-08 | F2 | Critical | Minor | yes | feedback | systemic_series | [2010 Market Flash Crash](https://incidentdatabase.ai/cite/28) |
| 29 | 2011-09-20 | F6 | Medium | n/a | yes | parallel | single | [Image Classification of Battle Tanks](https://incidentdatabase.ai/cite/29) |
| 30 | 2016-10-08 | F4 | Medium | Unclear/unknown | yes | serial | platform_repeat | [Poor Performance of Tesla Factory Robots](https://incidentdatabase.ai/cite/30) |
| 31 | 2017-12-03 | F5 | High | Negligible | no | none | single | [Driverless Train in Delhi Crashes due to Braking Failure](https://incidentdatabase.ai/cite/31) |
| 32 | 2017-09-13 | F6 | Medium | Negligible | yes | serial | platform_repeat | [Identical Twins Can Open Apple FaceID Protected Devices](https://incidentdatabase.ai/cite/32) |
| 33 | 2017-11-09 | F3 | Medium | Negligible | yes | serial | platform_repeat | [Amazon Alexa Plays Loud Music when Owner is Away](https://incidentdatabase.ai/cite/33) |
| 34 | 2015-12-05 | F3 | Medium | Negligible | yes | serial | platform_repeat | [Amazon Alexa Responding to Environmental Inputs](https://incidentdatabase.ai/cite/34) |
| 35 | 2014-10-18 | F1 | High | Negligible | yes | parallel | single | [Employee Automatically Terminated by Computer Program](https://incidentdatabase.ai/cite/35) |
| 36 | 2018-11-06 | F3 | Medium | Negligible | yes | parallel | single | [Picture of Woman on Side of Bus Shamed for Jaywalking](https://incidentdatabase.ai/cite/36) |
| 37 | 2016-08-10 | F6 | High | Negligible | yes | serial | platform_repeat | [Amazon’s Experimental Hiring Tool Allegedly Displayed Gender Bias in Candidate Rankings](https://incidentdatabase.ai/cite/37) |
| 38 | 2016-06-02 | F2 | Medium | Negligible | yes | parallel | single | [Game AI System Produces Imbalanced Game](https://incidentdatabase.ai/cite/38) |
| 39 | 2017-07-01 | F1 | Medium | Negligible | yes | parallel | single | [Deepfake Obama Introduction of Deepfakes](https://incidentdatabase.ai/cite/39) |
| 40 | 2016-05-23 | F6 | High | Minor | yes | parallel | single | [COMPAS Algorithm Reportedly Performs Poorly in Crime Recidivism Prediction](https://incidentdatabase.ai/cite/40) |
| 41 | 2018-04-02 | F1 | Low | Negligible | yes | serial | single | [All Image Captions Produced are Violent](https://incidentdatabase.ai/cite/41) |
| 42 | 1996-04-03 | F2 | Medium | Negligible | yes | serial | single | [Inefficiencies in the United States Resident Matching Program](https://incidentdatabase.ai/cite/42) |
| 43 | 1998-03-05 | F6 | High | Moderate | yes | parallel | single | [Racist AI behaviour is not a new problem](https://incidentdatabase.ai/cite/43) |
| 44 | 2008-07-01 | F1 | Medium | Negligible | yes | parallel | single | [Machine Personal Assistants Failed to Maintain Social Norms](https://incidentdatabase.ai/cite/44) |
| 45 | 2011-04-05 | F3 | Medium | Negligible | yes | serial | platform_repeat | [Defamation via AutoComplete](https://incidentdatabase.ai/cite/45) |
| 46 | 2014-01-21 | F5 | High | Negligible | no | none | single | [Nest Smoke Alarm Erroneously Stops Alarming](https://incidentdatabase.ai/cite/46) |
| 47 | 2016-09-06 | F6 | Medium | Negligible | no | none | single | [LinkedIn Search Prefers Male Names](https://incidentdatabase.ai/cite/47) |
| 48 | 2016-12-07 | F6 | Medium | Negligible | no | none | single | [Passport checker Detects Asian man's Eyes as Closed](https://incidentdatabase.ai/cite/48) |
| 49 | 2016-09-05 | F6 | Medium | Negligible | no | none | single | [AI Beauty Judge Did Not Like Dark Skin](https://incidentdatabase.ai/cite/49) |
| 50 | 2016-06-17 | F5 | High | Moderate | no | none | single | [The DAO Hack](https://incidentdatabase.ai/cite/50) |
