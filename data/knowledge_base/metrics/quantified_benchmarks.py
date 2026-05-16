"""
Quantified Metrics and Performance Benchmark Database
Measurable standards based on real industry data
"""

# System performance benchmark data
PERFORMANCE_BENCHMARKS = """
[System Performance Industry Benchmarks - 2024 Data]

Real-time System Response Time Benchmarks:

Hard Real-time Systems (Life-safety Critical):
- Automotive braking systems: ≤10ms
- Medical life support: ≤5ms  
- Aviation flight control systems: ≤2ms
- Nuclear reactor control: ≤1ms

Soft Real-time Systems (Business Critical):
- Financial trading systems: ≤50ms
- Telecommunications call setup: ≤500ms
- Video conferencing delay: ≤150ms
- Online gaming latency: ≤30ms

Throughput Benchmarks (TPS - Transactions Per Second):
- Visa payment network: 65,000 TPS (theoretical), 4,000 TPS (average)
- Alipay: 325,000+ TPS (Double 11 peak)
- Bitcoin network: 7 TPS
- Ethereum network: 15 TPS
- High-frequency trading systems: 1M+ TPS

Availability Benchmarks (Annual Downtime):
- 99.9% ("Three 9s"): 8.77 hours downtime
- 99.99% ("Four 9s"): 52.6 minutes downtime  
- 99.999% ("Five 9s"): 5.26 minutes downtime
- 99.9999% ("Six 9s"): 31.56 seconds downtime

Actual Achievement Rates (2023 Statistics):
- AWS: 99.99% (EC2 service)
- Google Cloud: 99.95% (Compute Engine)
- Microsoft Azure: 99.9% (Virtual Machines)
- Taobao: 99.99% (User-perceived availability)

Scalability Benchmarks:
- Horizontal scaling efficiency: Ideal linear scaling vs actual 60-80% efficiency
- Database sharding: Single shard ≤10TB, cross-shard query 30% performance loss
- Microservices architecture: Service count vs communication overhead (n² complexity)
- CDN cache hit rate: 85-95% considered good
"""

# Security quantified metrics
SECURITY_METRICS = """
[Security Quantified Metrics Benchmarks]

Cybersecurity Maturity Indicators:

Vulnerability Management Metrics:
- High-risk vulnerability fix time: ≤24 hours (industry standard)
- Medium-risk vulnerability fix time: ≤7 days
- Low-risk vulnerability fix time: ≤30 days
- Vulnerability scan frequency: Weekly automated + Quarterly manual
- Zero-day vulnerability response: ≤4 hours assessment, ≤12 hours mitigation measures

Intrusion Detection Metrics:
- False Positive Rate: ≤2% (excellent), ≤5% (acceptable)
- False Negative Rate: ≤0.1% (critical systems), ≤1% (general systems)
- Threat detection time: ≤1 hour (advanced threats), ≤15 minutes (known threats)
- Incident response time: ≤30 minutes confirmation, ≤2 hours initial response

Real Attack Statistics (2023 Global Data):
- Average data breach cost: $4.45M (IBM report)
- Average detection time: 277 days
- Average containment time: 70 days
- Ransomware attacks: 71% of organizations attacked
- Insider threat proportion: 34% of security incidents caused by insiders

Encryption Strength Standards:
- Symmetric encryption: AES-256 (government grade), AES-128 (commercial grade)
- Asymmetric encryption: RSA-2048 (minimum), RSA-3072 (recommended), ECC-256
- Hash algorithms: SHA-256 (recommended), SHA-3 (new standard)
- Key rotation: Symmetric keys ≤1 year, Asymmetric keys ≤2 years

Actual Deployment Statistics:
- HTTPS adoption rate: 95%+ (global websites)
- TLS 1.3 adoption rate: 60%+ (2024)
- Multi-factor authentication: 78% of enterprises enabled (Microsoft 2024 stats)
- Zero Trust architecture: 21% of enterprises fully deployed (Gartner 2024)
"""

# Energy efficiency and sustainability metrics
SUSTAINABILITY_METRICS = """
[Sustainability and Energy Efficiency Metrics Benchmarks]

Data Center Energy Efficiency Metrics:

PUE (Power Usage Effectiveness):
- Excellent level: PUE ≤ 1.2
- Good level: PUE ≤ 1.5  
- Industry average: PUE ≈ 1.8
- Traditional data centers: PUE ≥ 2.0

Real Cases (2024 Data):
- Google data centers: PUE 1.10 (annual average)
- Microsoft data centers: PUE 1.125
- Alibaba Cloud data centers: PUE 1.3 (Zhangbei data center)
- Facebook Prineville: PUE 1.07

WUE (Water Usage Effectiveness):
- Excellent level: WUE ≤ 0.5 L/kWh
- Acceptable level: WUE ≤ 1.8 L/kWh
- Waterless cooling data centers: WUE ≈ 0 (Facebook Prineville)

Carbon Emission Metrics:
- Data center carbon intensity: 0.5-0.9 kgCO2/kWh (coal power) vs 0.02-0.05 (renewable)
- Cloud service carbon efficiency: 77% more efficient (vs enterprise self-built data centers)
- Renewable energy usage rate: Google 100%, Microsoft 60%, AWS 50%

Software Energy Efficiency Metrics:
- Algorithm carbon footprint: GPT-3 training emitted 552 tons CO2
- Video streaming: 4.6gCO2 per hour (mobile) vs 4.3kgCO2 (large screen)
- Bitcoin mining: Annual power consumption 143TWh (equivalent to Argentina)
- Web page loading energy: Average 4.6g CO2 per visit

Device Lifecycle:
- Server average lifespan: 4-6 years (cloud providers) vs 3-4 years (enterprise)
- Mobile device replacement cycle: 2.7 years (global average)
- E-waste recycling rate: 20% (global) vs 45% (EU)
- Rare earth element recycling rate: <1% (critical materials)

Actual Green IT Investment:
- Global green IT spending: Expected $263 billion in 2024
- Enterprise sustainable IT budget share: Average 6% (Gartner 2024)
- ROI payback period: Green IT investment average 2.5 years payback
- Regulatory driven: 42% of investment driven by regulatory compliance
"""

# User experience and usability metrics
UX_USABILITY_METRICS = """
[User Experience Quantified Metrics Benchmarks]

Website Performance Metrics (Core Web Vitals):

LCP (Largest Contentful Paint):
- Excellent: ≤2.5 seconds
- Needs improvement: 2.5-4.0 seconds
- Poor: >4.0 seconds
- Industry average: 3.2 seconds (mobile), 2.1 seconds (desktop)

FID (First Input Delay):
- Excellent: ≤100 milliseconds
- Needs improvement: 100-300 milliseconds  
- Poor: >300 milliseconds
- Industry average: 25 milliseconds (mobile), 12 milliseconds (desktop)

CLS (Cumulative Layout Shift):
- Excellent: ≤0.1
- Needs improvement: 0.1-0.25
- Poor: >0.25
- Industry average: 0.15 (mobile), 0.08 (desktop)

Mobile App Performance:
- App startup time: ≤2 seconds (cold start), ≤1 second (warm start)
- Crash rate: ≤0.1% (excellent), ≤1% (acceptable)
- ANR rate: ≤0.05% (Android Application Not Responding)
- Memory usage: ≤100MB (lightweight apps), ≤500MB (heavy apps)

User Satisfaction Metrics:
- NPS (Net Promoter Score): >50 (excellent), 0-50 (good), <0 (needs improvement)
- CSAT (Customer Satisfaction): >85% (excellent), 70-85% (good)
- User retention rate: Day 1: 75%, Day 7: 35%, Day 30: 20% (mobile app average)
- Average session duration: 2-4 minutes (content apps), 8-12 minutes (gaming apps)

Real Industry Data (2024):
- E-commerce conversion rate: Average 2.86% (desktop), 1.53% (mobile)
- Shopping cart abandonment rate: Average 69.99%
- Page bounce rate: Average 53% (all industries)
- Search success rate: Average 73% (enterprise website internal search)

Accessibility Metrics:
- WCAG 2.1 AA compliance: 71.4% of websites compliant (WebAIM 2024 survey)
- Automated testing coverage: Can only detect 20-30% of accessibility issues
- Manual assessment cost: $15,000-50,000 (large websites)
- Legal risk: US ADA lawsuits growing 15% annually
"""

# Summary of quantified metrics knowledge
QUANTIFIED_METRICS = {
    "performance_benchmarks": PERFORMANCE_BENCHMARKS,
    "security_metrics": SECURITY_METRICS,
    "sustainability_metrics": SUSTAINABILITY_METRICS,
    "ux_usability_metrics": UX_USABILITY_METRICS,
}
