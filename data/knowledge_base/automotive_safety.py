"""
Automotive Safety Domain Professional Knowledge
Based on ISO 26262 Functional Safety Standards and Actual Engineering Experience
"""

# ISO 26262 ASIL Levels and Specific Requirements
ISO_26262_KNOWLEDGE = """
[ISO 26262 Automotive Functional Safety Standard]

ASIL Level Assignment Principles:
- Severity(S) × Exposure(E) × Controllability(C) = ASIL Level

ASIL D (Most Stringent):
- Failure rate requirement: < 10^-8 /hour
- Applicable systems: Braking system, steering system, airbags
- Development requirements: Formal methods, comprehensive verification, independent confirmation

ASIL C:
- Failure rate requirement: < 10^-7 /hour
- Applicable systems: Collision warning, lane keeping assist
- Development requirements: Structured methods, rigorous testing, semi-independent confirmation

ASIL B:
- Failure rate requirement: < 10^-7 /hour
- Applicable systems: Headlight control, windshield wiper system
- Development requirements: Semi-formal methods, high coverage testing

ASIL A:
- Failure rate requirement: < 10^-6 /hour
- Applicable systems: Entertainment system, air conditioning system
- Development requirements: Structured development, basic testing

Specific KAOS Goal Mapping:
High-level safety goal: "Vehicle should avoid causing casualties"
└── ASIL D sub-goal: "Braking system failure rate < 10^-8/hour"
    └── Design requirement: "Dual redundant braking circuits + fault detection < 100ms"
        └── Implementation measures: "Primary braking system + backup braking + brake assist"
"""

# Autonomous Driving L0-L5 Level Safety Requirements
AUTONOMOUS_DRIVING_LEVELS = """
[Autonomous Driving Safety Level Requirements]

L0 Level (No Automation):
- Safety requirements: Traditional automotive safety standards
- Driver responsibility: 100% driving tasks
- System responsibility: Basic warnings and alerts

L1 Level (Driver Assistance):
- Safety requirements: Driver can immediately take over when assistance functions fail
- System examples: Adaptive Cruise Control (ACC)
- KAOS goal: "Alert driver within 2 seconds when system fails"
- Failure rate: < 10^-6 /hour (ASIL B)

L2 Level (Partial Automation):
- Safety requirements: Driver continuous monitoring + system monitors driver state
- System examples: Tesla Autopilot, GM Super Cruise
- KAOS goals:
  - "Detect driver distraction < 3 seconds"
  - "Safely exit automatic mode when lane lines are lost"
- Failure rate: < 10^-7 /hour (ASIL C)

L3 Level (Conditional Automation):
- Safety requirements: System responsible within ODD + takeover request response time limits
- System examples: Audi A8 Traffic Jam Pilot
- KAOS goals:
  - "ODD boundary detection accuracy > 99.9%"
  - "Takeover request to driver response < 10 seconds"
  - "Minimal risk state when takeover fails < 30 seconds"
- Failure rate: < 10^-8 /hour (ASIL D)

L4 Level (High Automation):
- Safety requirements: Complete system responsibility within ODD + minimal risk state capability
- System examples: Waymo, Cruise (defined areas)
- KAOS goals:
  - "All safety-critical functions with redundant design"
  - "Safe parking success rate when failed > 99.99%"
  - "Passenger emergency intervention channel < 5 second activation"

L5 Level (Full Automation):
- Safety requirements: System completely responsible for all scenarios
- Current status: Theoretical stage, no commercial products
- KAOS goal: "All-scenario safety equivalent to professional drivers"
"""

# Real Automotive Safety Failure Cases and Lessons
AUTOMOTIVE_FAILURE_CASES = """
[Automotive Safety Failure Case Analysis]

Case 1: Toyota Unintended Acceleration Incident (2009-2011)
Root cause: Software defects + insufficient design redundancy
Lessons learned:
- KAOS modeling missing: Failed to consider abnormal interactions between accelerator pedal and software
- Incomplete safety goals: "Prevent unintended acceleration" goal lacked specific failure mode analysis
- Insufficient verification: Software testing did not cover boundary conditions like electromagnetic interference

Corrected KAOS Goals:
G1: Prevent vehicle unintended acceleration
├── G1.1: Throttle input validation and limitation
│   ├── G1.1.1: Throttle signal reasonableness check (0-100% range)
│   ├── G1.1.2: Throttle change rate limitation (<30%/second)
│   └── G1.1.3: Throttle pedal position sensor redundancy
├── G1.2: Brake priority control
│   ├── G1.2.1: Brake signal overrides throttle signal
│   └── G1.2.2: Brake pedal detection <50ms response
└── G1.3: Emergency shutdown mechanism
    ├── G1.3.1: Driver can manually cut power
    └── G1.3.2: Automatic power limitation when system abnormal

Case 2: Tesla Model S Collision Accident (2016)
Root cause: Sensor limitations + algorithm defects + user misuse
Lessons learned:
- Insufficient sensor fusion: Relying only on cameras failed to detect white trailer
- Vague ODD definition: Did not clearly restrict to highway scenarios
- Insufficient user education: Driver over-reliance on system

Corrected KAOS Goals:
G1: Autonomous driving system safe operation
├── G1.1: Multi-sensor fusion detection
│   ├── G1.1.1: Camera + radar + LiDAR redundancy
│   ├── G1.1.2: Sensor failure detection <100ms
│   └── G1.1.3: Severe weather degradation strategy
├── G1.2: ODD boundary management
│   ├── G1.2.1: Real-time road condition adaptability assessment
│   ├── G1.2.2: Safe handover when exiting ODD
│   └── G1.2.3: Conservative handling of boundary conditions
└── G1.3: Human-machine interaction safety
    ├── G1.3.1: Driver state monitoring
    ├── G1.3.2: System capability boundary reminders
    └── G1.3.3: Takeover training and validation
"""

# Automotive Cybersecurity Requirements (Based on ISO/SAE 21434)
AUTOMOTIVE_CYBERSECURITY = """
[Automotive Cybersecurity Requirements] ISO/SAE 21434

Cybersecurity Goal Hierarchy:
High-level goal: "Protect vehicle from cyber attacks"
├── Confidentiality: Sensitive information not accessed by unauthorized parties
├── Integrity: Data and system functions not maliciously modified
├── Availability: Critical functions remain available during attacks

Specific KAOS Security Goals:

G1: Vehicle network communication security
├── G1.1: CAN bus message authentication
│   ├── G1.1.1: Critical message digital signatures (braking, steering)
│   ├── G1.1.2: Message freshness verification (timestamp + sequence number)
│   └── G1.1.3: Abnormal message detection and isolation
├── G1.2: ECU identity verification
│   ├── G1.2.1: ECU legitimacy verification at startup
│   ├── G1.2.2: Runtime ECU status monitoring
│   └── G1.2.3: Suspicious ECU network isolation
└── G1.3: Network traffic monitoring
    ├── G1.3.1: Abnormal traffic pattern detection
    ├── G1.3.2: Intrusion Detection System (IDS) deployment
    └── G1.3.3: Security event logging

G2: External connection security
├── G2.1: Cellular network connection protection
│   ├── G2.1.1: End-to-end encrypted communication
│   ├── G2.1.2: Server identity verification
│   └── G2.1.3: Malicious server detection
├── G2.2: WiFi connection security management
│   ├── G2.2.1: Trusted WiFi network whitelist
│   ├── G2.2.2: Open WiFi connection restrictions
│   └── G2.2.3: WiFi traffic isolation
└── G2.3: OTA update security
    ├── G2.3.1: Update package digital signature verification
    ├── G2.3.2: Update rollback mechanism
    └── G2.3.3: Update process status monitoring

Threat Assessment Levels:
- High risk: Braking/steering systems under attack → ASIL D safety requirements
- Medium risk: Infotainment system exploited as springboard → ASIL B requirements
- Low risk: Non-critical function data leakage → ASIL A requirements
"""

AUTOMOTIVE_KNOWLEDGE = {
    "iso_26262": ISO_26262_KNOWLEDGE,
    "autonomous_levels": AUTONOMOUS_DRIVING_LEVELS,
    "failure_cases": AUTOMOTIVE_FAILURE_CASES,
    "cybersecurity": AUTOMOTIVE_CYBERSECURITY,
}
