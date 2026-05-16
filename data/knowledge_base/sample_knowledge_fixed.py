"""
OpenRE-Bench / QUARE Professional Knowledge Base - Enhanced Version
Contains KAOS modeling cases, best practices, domain expertise, and negotiation patterns
Based on latest 2024 industry standards and real industrial cases
"""

# Import new professional knowledge modules
try:
    from .domains.automotive_safety import AUTOMOTIVE_PROFESSIONAL_KNOWLEDGE
    from .cases.real_world_studies import REAL_WORLD_CASES
    from .standards.latest_2024 import LATEST_STANDARDS_2024
    from .metrics.quantified_benchmarks import QUANTIFIED_METRICS

    # Integrate all professional knowledge
    EXTENDED_KNOWLEDGE = {
        **AUTOMOTIVE_PROFESSIONAL_KNOWLEDGE,
        **REAL_WORLD_CASES,
        **LATEST_STANDARDS_2024,
        **QUANTIFIED_METRICS,
    }

except ImportError as e:
    # If new modules don't exist, use empty dictionary
    print(f"Warning: Could not import extended knowledge: {e}")
    EXTENDED_KNOWLEDGE = {}

# Original basic knowledge remains unchanged
# Real KAOS modeling cases
KAOS_CASE_STUDIES = """
[Autonomous Driving System KAOS Case]

Strategic Level Goals:
- G1: Provide safe autonomous driving service
- G2: Ensure efficient traffic flow
- G3: Achieve environmentally friendly transportation
- G4: Build user trust relationships
- G5: Fulfill social responsibility obligations

Tactical Level Refinement:
- G1.1: Detect and avoid collision threats
- G1.2: Comply with traffic regulations and standards
- G1.3: Safely park in abnormal situations
- G2.1: Optimize path planning algorithms
- G2.2: Reduce unnecessary braking and acceleration
- G3.1: Minimize energy consumption
- G3.2: Reduce exhaust emissions

Operational Level Implementation:
- G1.1.1: Radar system continuously scans surrounding environment
- G1.1.2: Camera identifies pedestrians and obstacles
- G1.1.3: Emergency braking system responds within 0.3 seconds
- G2.1.1: GPS navigation system provides real-time traffic conditions
- G2.1.2: Machine learning algorithms predict traffic patterns
"""

# Domain-specific best practices
DOMAIN_BEST_PRACTICES = """
[Automotive Safety Domain] ISO 26262 Best Practices:

ASIL Level Assignment:
- ASIL D (Highest): Braking system, steering system failure
- ASIL C: Airbag, collision warning failure  
- ASIL B: Lighting system, windshield wiper failure
- ASIL A (Lowest): Entertainment system, air conditioning failure

Safety Goal Template:
"Under [operating conditions], when [failure event] occurs, the system shall [safety response], with probability not exceeding [ASIL requirement]"

Example:
"During high-speed driving, when main braking system fails, backup braking system shall activate within 1 second, with failure probability <10^-8/hour (ASIL D)"

[Medical Device Domain] IEC 62304 Best Practices:

Safety Classification:
- Class C: Software failure may cause death or serious injury
- Class B: Software failure may cause non-serious injury
- Class A: Software failure will not cause injury

Risk Control Measures:
1. Software Architecture Control (design redundancy)
2. Software Verification Control (rigorous testing)
3. Process Control (development process standards)
"""

# Quantifiable quality attribute metrics
QUALITY_METRICS = """
[Quantifiable Quality Attribute Metrics]

Safety Metrics:
- MTBF (Mean Time Between Failures): >10^6 hours
- Fault Detection Coverage: >99.9%
- Safety Integrity Level: SIL 1-4
- Hazardous Event Probability: <10^-6 ~ 10^-9 /hour

Efficiency Metrics:
- Response Time: <100ms (real-time systems)
- Throughput: >1000 TPS
- CPU Utilization: <80%
- Memory Usage: <512MB

Availability Metrics:
- System Uptime: >99.99% (annual downtime <52 minutes)
- Recovery Time Objective (RTO): <15 minutes
- Recovery Point Objective (RPO): <1 hour
- Mean Time To Repair (MTTR): <4 hours

Sustainability Metrics:
- Energy Efficiency: <50W power consumption
- Carbon Footprint: <100kg CO2/year
- Resource Utilization: >90%
- Component Recyclability: >85%

Trustworthiness Metrics:
- Accuracy: >95%
- Explainability Score: >0.8
- Audit Trail Integrity: 100%
- User Satisfaction: >4.5/5.0
"""

# Real negotiation conflict patterns and solutions
NEGOTIATION_PATTERNS = """
[Real Negotiation Conflict Patterns]

Pattern 1: Safety vs Efficiency Conflict
Scenario: In autonomous driving systems, safety detection requires 300ms, but real-time response requires <100ms
Resolution Strategies:
- Layered Detection: Fast detection (50ms) + Deep verification (250ms in parallel)
- Progressive Response: Immediate warning + Delayed precise action
- Risk Assessment: Lower detection precision in low-risk scenarios, higher precision in high-risk scenarios

Pattern 2: Environmental Friendliness vs Performance Requirements Conflict  
Scenario: Energy-saving mode reduces CPU frequency affecting computational performance
Resolution Strategies:
- Smart Frequency Scaling: Dynamically adjust based on task importance
- Load Prediction: Enter high-performance mode in advance
- Hybrid Strategy: High performance for critical tasks, energy saving for background tasks

Pattern 3: Trustworthiness vs Efficiency Conflict
Scenario: Decision explanation generation requires additional 50% computation time
Resolution Strategies:
- Asynchronous Explanation: Execute decision first, generate explanation later
- Tiered Explanation: Brief explanation generated in real-time, detailed explanation offline
- User Choice: Allow users to trade off between speed and transparency

Negotiation Assessment Metrics:
- Conflict Resolution Rate: Resolved conflicts / Total conflicts
- Compromise Quality: Σ(Agent satisfaction scores) / Number of agents
- Resolution Time Efficiency: Average time per conflict resolution
- Solution Stability: Rate of solution retention in subsequent rounds
"""

# Industry standards and compliance requirements
COMPLIANCE_KNOWLEDGE = """
[Industry Standards Compliance Requirements]

Automotive Industry:
- ISO 26262: Functional safety management, V-model development process
- ISO 21448: SOTIF (Safety of the Intended Functionality)
- UN-R157: Automated Lane Keeping System certification
- NHTSA: National Highway Traffic Safety Administration requirements

Aerospace:
- DO-178C: Software considerations in airborne systems
- DO-254: Design assurance guidance for airborne electronic hardware
- ARP4754A: Certification considerations for civil aircraft and systems
- RTCA DO-326A: Cybersecurity design and verification

Medical Devices:
- IEC 62304: Medical device software lifecycle processes
- ISO 14971: Medical devices risk management
- FDA 510(k): US FDA device certification process
- MDR: European Union Medical Device Regulation

Financial Technology:
- PCI DSS: Payment Card Industry Data Security Standard
- SOX: Sarbanes-Oxley Act
- Basel III: Banking regulatory framework
- GDPR: General Data Protection Regulation

KAOS Integration Points for Each Standard:
1. Standard requirements → High-level safety goals
2. Compliance clauses → Tactical-level constraint conditions
3. Specific specifications → Operational-level technical measures
4. Audit requirements → Traceability relationships
"""

# Error patterns and lessons learned
ERROR_PATTERNS = """
[Common KAOS Modeling Errors and Corrections]

Error 1: Goals too abstract and unverifiable
Error Example: "System should be safe"
Correction Example: "System shall stop within 2 seconds after detecting obstacles, with success rate >99.9%"

Error 2: Missing key stakeholders
Error Example: Only considering users and system, ignoring regulatory authorities
Correction Example: Include users, system, regulatory authorities, maintenance personnel, environment

Error 3: Lack of traceability between goals
Error Example: High-level goals cannot correspond to low-level implementations
Correction Example: Establish clear refinement chains G1→G1.1→G1.1.1

Error 4: Ignoring exceptions and boundary conditions
Error Example: Only considering normal operation flows
Correction Example: Include fault handling, degraded modes, emergency situations

Error 5: Unresolved quality attribute conflicts
Error Example: Simultaneously requiring highest safety and highest performance
Correction Example: Establish trade-off strategies and dynamic adjustment mechanisms
"""

# Safety-related knowledge
SAFETY_KNOWLEDGE = """
Safety Requirements Engineering Key Elements:

Risk Identification:
- Systematic Hazard Analysis (HAZOP)
- Failure Mode and Effects Analysis (FMEA)
- Fault Tree Analysis (FTA)

Safety Goal Hierarchy:
- System-level safety goals
- Subsystem safety requirements
- Component safety measures

Critical Safety Attributes:
- Reliability
- Availability  
- Integrity
- Confidentiality

Safety Standards:
- ISO 26262 (Automotive functional safety)
- IEC 61508 (Functional safety foundation standard)
- DO-178C (Aviation software)
"""

# Efficiency-related knowledge
EFFICIENCY_KNOWLEDGE = """
Efficiency Optimization Key Areas:

Performance Metrics:
- Response Time
- Throughput
- Resource Utilization
- Scalability

Optimization Strategies:
- Algorithm optimization
- Data structure selection
- Caching mechanisms
- Parallel processing
- Load balancing

Efficiency Trade-offs:
- Time vs space complexity
- Accuracy vs speed
- Real-time vs batch processing
- Local vs distributed processing
"""

# Sustainability-related knowledge
SUSTAINABILITY_KNOWLEDGE = """
Green Software Engineering Principles:

Energy Efficiency:
- Computing resource optimization
- Data transmission minimization
- Idle resource release
- Smart power management

Environmental Impact:
- Carbon footprint assessment
- Life cycle analysis
- Renewable energy usage
- Waste reduction

Sustainable Design:
- Modular architecture
- Reusable components
- Long-term maintainability
- Technical debt management
"""

# Trustworthiness-related knowledge
TRUSTWORTHINESS_KNOWLEDGE = """
Trustworthy System Construction Elements:

Reliability Assurance:
- Fault-tolerant design
- Redundancy mechanisms
- Self-healing capabilities
- Monitoring and alerting

Transparency Requirements:
- Decision explainability
- Audit trails
- State observability
- User feedback

Verification Methods:
- Formal verification
- Test coverage
- Peer review
- User acceptance
"""

# Responsibility-related knowledge
RESPONSIBILITY_KNOWLEDGE = """
Responsibility Design Principles:

Accountability Mechanisms:
- Decision recording
- Responsibility tracing
- Error attribution
- Remedial measures

Ethical Considerations:
- Fairness principles
- Privacy protection
- Non-discriminatory design
- Social impact assessment

Compliance Requirements:
- Legal and regulatory compliance
- Industry standard conformity
- Regulatory requirement satisfaction
- Social responsibility fulfillment
"""

# Multi-agent negotiation knowledge
NEGOTIATION_KNOWLEDGE = """
Multi-agent Negotiation Strategies:

Negotiation Protocols:
- Contract Net Protocol
- Alternating Offers Protocol
- Auction-based Negotiation
- Argumentation Framework

Conflict Resolution:
- Compromise strategies
- Voting mechanisms
- Arbitration schemes
- Priority ordering

Negotiation Evaluation:
- Pareto optimality
- Nash equilibrium
- Social welfare maximization
- Fairness metrics
"""

# Summary of all professional knowledge - Enhanced version
PROFESSIONAL_KNOWLEDGE = {
    # Basic knowledge
    "kaos_case_studies": KAOS_CASE_STUDIES,
    "domain_best_practices": DOMAIN_BEST_PRACTICES,
    "quality_metrics": QUALITY_METRICS,
    "negotiation_patterns": NEGOTIATION_PATTERNS,
    "compliance_knowledge": COMPLIANCE_KNOWLEDGE,
    "error_patterns": ERROR_PATTERNS,
    "safety_engineering": SAFETY_KNOWLEDGE,
    "efficiency_optimization": EFFICIENCY_KNOWLEDGE,
    "sustainability": SUSTAINABILITY_KNOWLEDGE,
    "trustworthiness": TRUSTWORTHINESS_KNOWLEDGE,
    "responsibility": RESPONSIBILITY_KNOWLEDGE,
    "negotiation": NEGOTIATION_KNOWLEDGE,
    # Extended professional knowledge
    **EXTENDED_KNOWLEDGE,
}
