#!/usr/bin/env python3
"""
Knowledge Base Content Management Guide
Explains what types of content should be placed in the knowledge base and how to organize it
"""


def knowledge_base_guide():
    """Knowledge base content design guide"""

    guide = """
OpenRE-Bench / QUARE Knowledge Base Content Design Guide
================================================

📚 Types of content that should be included:

1. [Real Cases and Templates] - Most Important!
   ✅ Real KAOS model examples (autonomous driving, medical devices, financial systems)
   ✅ Successful requirement decomposition patterns (strategic→tactical→operational)
   ✅ Specific quantified indicators for quality attributes
   ✅ Validated goal-requirement mapping relationships

2. [Domain Professional Knowledge]
   ✅ Industry standards and compliance requirements (ISO 26262, IEC 62304, etc.)
   ✅ Domain-specific safety/performance/sustainability best practices
   ✅ Common failure modes and prevention measures
   ✅ Specific requirements from regulatory bodies

3. [Negotiation and Conflict Resolution Patterns]
   ✅ Real quality attribute conflict cases
   ✅ Validated negotiation strategies
   ✅ Assessment methods for compromises and trade-offs
   ✅ Convergence patterns for multi-round negotiations

4. [Error Patterns and Lessons Learned]
   ✅ Common KAOS modeling errors
   ✅ Typical pitfalls in requirement analysis
   ✅ Checklists for quality attribute omissions
   ✅ Methods to fix traceability gaps

5. [Quantitative Indicators and Measurement Standards]
   ✅ Testable quality attribute metrics
   ✅ Industry-recognized performance benchmarks
   ✅ Quantitative methods for risk assessment
   ✅ Criteria for judging negotiation success

❌ Content that should NOT be included:

❌ Overly abstract theoretical knowledge
❌ Generic information unrelated to KAOS modeling
❌ Outdated standards and specifications
❌ Unverifiable subjective judgments
❌ Overly specific implementation code

🎯 Core Value of Knowledge Base:

1. [Enhance Agent Professionalism]
   - Safety agents can reference specific clauses from ISO 26262
   - Efficiency agents can refer to industry performance benchmarks
   - Sustainability agents can use carbon footprint calculation methods

2. [Improve Negotiation Quality]
   - Provide real conflict resolution cases as references
   - Use validated compromise strategies
   - Make decisions based on historical success patterns

3. [Ensure Compliance]
   - Automatically check for missing regulatory requirements
   - Remind about critical safety/privacy considerations
   - Ensure generated KAOS models comply with industry standards

4. [Accelerate Learning Process]
   - Rapid knowledge acquisition in new domains
   - Avoid repeating known error patterns
   - Reuse successful analysis templates

🔧 Practical Application Scenarios of Knowledge Base:

Scenario 1: Autonomous Driving System Analysis
Query: "Safety requirements for autonomous driving emergency braking"
Returns: ISO 26262 ASIL level requirements + specific response time indicators + failure rate limits

Scenario 2: Performance vs Safety Trade-offs
Query: "Conflicts between safety detection and response time in real-time systems"
Returns: Layered detection strategies + risk assessment methods + success cases

Scenario 3: Sustainability Quantification
Query: "Methods for measuring energy efficiency in software systems"
Returns: Specific power consumption indicators + measurement tools + industry benchmarks

📈 Metrics to Measure Knowledge Base Effectiveness:

1. Retrieval Precision: Proportion of query-relevant results
2. Application Frequency: Number of times knowledge is referenced by agents
3. Negotiation Improvement: Quality improvement in conflict resolution after enabling RAG
4. Model Quality: Completeness and professionalism of generated KAOS models
5. User Satisfaction: Credibility and practicality of system output
"""

    return guide


def example_knowledge_structure():
    """Demonstrate the ideal structure of knowledge base"""

    structure = """
🗂️ Recommended Knowledge Base File Structure:

knowledge_base/
├── domains/                    # Domain-specific knowledge
│   ├── automotive_safety.py   # Automotive safety domain
│   ├── medical_devices.py     # Medical devices domain
│   ├── financial_systems.py   # Financial systems domain
│   └── aerospace.py           # Aerospace domain
│
├── standards/                  # Industry standards
│   ├── iso_26262.py           # Automotive functional safety
│   ├── iec_62304.py           # Medical software standard
│   ├── do_178c.py             # Aviation software standard
│   └── common_criteria.py     # Information security common criteria
│
├── patterns/                   # Patterns and templates
│   ├── kaos_templates.py      # KAOS modeling templates
│   ├── negotiation_patterns.py # Negotiation patterns
│   ├── conflict_resolution.py  # Conflict resolution strategies
│   └── quality_tradeoffs.py    # Quality trade-off patterns
│
├── metrics/                    # Quantitative indicators
│   ├── safety_metrics.py      # Safety metrics
│   ├── performance_metrics.py # Performance metrics
│   ├── sustainability_metrics.py # Sustainability metrics
│   └── trust_metrics.py       # Trustworthiness metrics
│
└── cases/                      # Real cases
    ├── autonomous_driving.py  # Autonomous driving cases
    ├── medical_monitoring.py  # Medical monitoring cases
    ├── smart_grid.py         # Smart grid cases
    └── fintech_platform.py   # FinTech platform cases

Each file contains:
- Structured professional knowledge
- Referenceable specific indicators
- Real case studies
- Validated best practices
"""

    return structure


if __name__ == "__main__":
    print(knowledge_base_guide())
    print("\n" + "=" * 50 + "\n")
    print(example_knowledge_structure())

    print("\n💡 Next Step Recommendations:")
    print("1. Based on your main application domain, prioritize filling relevant knowledge")
    print("2. Add more real cases and quantitative indicators")
    print("3. Regularly update industry standards and best practices")
    print("4. Validate the improvement effect of knowledge base on agent performance through experiments")
