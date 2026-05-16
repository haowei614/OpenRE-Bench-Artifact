"""
Automotive Safety Domain Professional Knowledge Base
Based on ISO 26262 and real automotive industry experience
"""

# ISO 26262 detailed knowledge
ISO_26262_KNOWLEDGE = """
[ISO 26262 Automotive Functional Safety Standard]

ASIL Level Detailed Classification:

ASIL D (Most Stringent):
- Applicable systems: Braking systems, steering systems, airbag control
- Failure rate requirement: <10^-8 failures/hour
- Development process: V-model, 100% requirements traceability
- Verification requirements: Formal verification + fault injection testing
- Documentation requirements: Complete safety case, independent assessment

ASIL C:
- Applicable systems: Lane keeping assist, collision warning
- Failure rate requirement: <10^-7 failures/hour
- Development process: V-model, semi-formal methods
- Verification requirements: Structured test coverage
- Documentation requirements: Safety analysis documentation

ASIL B:
- Applicable systems: External lighting systems, windshield wiper control
- Failure rate requirement: <10^-6 failures/hour
- Development process: Structured development
- Verification requirements: Functional testing + integration testing

ASIL A (Most Lenient):
- Applicable systems: Entertainment systems, comfort functions
- Failure rate requirement: <10^-5 failures/hour
- Development process: Standard software development practices

Functional Safety Concept Development Process:
1. Hazard Analysis and Risk Assessment (HARA)
2. Functional Safety Concept (FSC)
3. Technical Safety Concept (TSC) 
4. System design and implementation
5. Safety verification and validation

Key Safety Mechanisms:
- Diagnostic Coverage: DC ≥99% (ASIL D)
- Fault Tolerant Time Interval: FTTI ≤100ms
- Safe state transition time: ≤2s
- Hardware Architecture Metrics: SPFM ≥99%, LFM ≤10%

Actual Quantified Metrics:
- Emergency braking response: Perception to execution ≤200ms
- Steering system fault detection: ≤50ms
- Airbag deployment decision: ≤10ms
- Engine management failure mode: Limp-home mode ≤3s
"""

# Autonomous driving safety architecture
AUTONOMOUS_DRIVING_SAFETY = """
[Autonomous Driving System Safety Architecture]

SAE L3+ Autonomous Driving Safety Requirements:

Perception System Redundancy:
- Primary sensors: LiDAR (360°, 100m range, ±5cm accuracy)
- Auxiliary sensors: 8 cameras + 12 millimeter-wave radars
- Backup sensors: Ultrasonic sensor array
- Fault detection time: ≤100ms
- Sensor fusion latency: ≤50ms

Decision System Architecture:
- Main controller: ASIL D level, dual-core lockstep
- Supervisory controller: ASIL D level, independent hardware
- Human takeover time: ≤8s (urban), ≤10s (highway)
- Minimal risk maneuver: Safe parking ≤15s

Execution System Requirements:
- Braking system: Redundant hydraulic + electronic braking
- Steering system: EPS + mechanical backup
- Powertrain system: Torque limiting + safe mode
- Communication system: V2X + 4G/5G dual communication

Safety Performance Metrics:
- Overall system failure rate: <10^-8/hour
- Collision avoidance success rate: >99.99%
- False positive rate: <0.1% (emergency braking)
- False negative rate: <0.01% (hazardous obstacles)

Weather Condition Adaptability:
- Clear weather: Full functionality operation
- Light rain (<10mm/h): 20% performance degradation
- Moderate rain (10-25mm/h): 50% performance degradation, speed limit
- Heavy rain (>25mm/h): Request human takeover
- Snow (visibility <50m): Safe parking
- Fog (visibility <100m): Speed limit 30km/h
"""

# Automotive cybersecurity
AUTOMOTIVE_CYBERSECURITY = """
[Automotive Cybersecurity - ISO/SAE 21434]

Cybersecurity Risk Levels:

Risk Level 1 (High Risk):
- Impact: Life safety threats
- Assets: Braking, steering, safety systems
- Protection measures: Hardware Security Module (HSM) + end-to-end encryption
- Monitoring requirements: Real-time anomaly detection ≤10ms

Risk Level 2 (Medium Risk):
- Impact: Vehicle function impact
- Assets: Engine control, transmission
- Protection measures: Digital signatures + access control
- Monitoring requirements: Periodic integrity checks ≤1s

Risk Level 3 (Low Risk):  
- Impact: User experience impact
- Assets: Entertainment systems, air conditioning
- Protection measures: Basic encryption + user authentication
- Monitoring requirements: Regular security scans

Specific Security Measures:
- Secure boot: Verify all ECU firmware signatures
- Secure communication: CAN-SEC protocol, AES-256 encryption
- Intrusion detection: Behavioral analysis + abnormal traffic detection
- Security updates: OTA updates, digital signature verification
- Key management: PKI infrastructure, key rotation cycle 30 days

Cybersecurity Testing:
- Penetration testing frequency: Quarterly
- Vulnerability scanning: Weekly automated scanning
- Emergency response time: Within 4 hours of discovery
- Patch deployment: Critical vulnerabilities 24 hours, general vulnerabilities 7 days
"""

# Real test data and benchmarks
AUTOMOTIVE_BENCHMARKS = """
[Automotive Industry Real Performance Benchmarks]

Braking System Benchmark Data:
- Dry road surface (μ=0.9): 60→0 km/h ≤39m
- Wet road surface (μ=0.7): 60→0 km/h ≤56m  
- Snow road surface (μ=0.3): 60→0 km/h ≤130m
- ABS response time: ≤15ms
- Braking force adjustment accuracy: ±5%

Steering System Benchmarks:
- Steering response time: ≤100ms
- Steering angle accuracy: ±0.5°
- Centering capability: Return to center ±2° within 3s
- Maximum torque when power assist fails: ≤60Nm

Engine Management Benchmarks:
- Idle stability: ±25 rpm
- Acceleration response (0-100 km/h): 6-15s depending on vehicle type
- Emission control: Euro 6 standard
- Fuel efficiency: NEDC cycle test results

Battery Management System (EV/HEV):
- Battery temperature control: 20-35°C ±2°C
- Charging efficiency: >95% (slow charging), >90% (fast charging)
- Battery capacity degradation: <20% (8 years/160,000 km)
- SOC estimation accuracy: ±5%

Connected Vehicle Performance:
- V2V communication latency: ≤100ms
- V2I communication range: ≥300m
- Data transmission reliability: >99.9%
- Positioning accuracy: ≤30cm (enhanced GPS)
"""

# Summary of automotive domain knowledge
AUTOMOTIVE_PROFESSIONAL_KNOWLEDGE = {
    "iso_26262_detailed": ISO_26262_KNOWLEDGE,
    "autonomous_driving_safety": AUTONOMOUS_DRIVING_SAFETY,
    "automotive_cybersecurity": AUTOMOTIVE_CYBERSECURITY,
    "automotive_benchmarks": AUTOMOTIVE_BENCHMARKS,
}
