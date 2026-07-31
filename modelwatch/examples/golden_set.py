"""Golden question/answer set for Sanad's chatbot, used as ModelWatch's
LLMAdapter baseline (see sanad_golden_set_runner.py).

Every expected_answer here is grounded in the actual text of the sample
contract it's paired with (verified by reading the extracted/chunked
text directly, not written from assumption) -- see each source_file.

Coverage: rental and freelance/service contracts, drawn from India-specific
templates. Employment has no pairs yet: the only file dropped into
sample_docs/employment/ turned out to be a compilation of real people's
real offer letters (not a template) and was excluded from the project
entirely -- see project history. Once a proper blank employment template
is added to sample_docs/employment/, add its pairs here following the
same pattern.
"""
from __future__ import annotations

from typing import TypedDict


class GoldenPair(TypedDict):
    contract_type: str
    source_file: str
    prompt: str
    expected_answer: str


GOLDEN_SET: list[GoldenPair] = [
    # -- rental_agreement_sample_1.pdf (OWNER / TENANT residential rental) --
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample_1.pdf",
        "prompt": "When is the monthly rent due?",
        "expected_answer": "The rent shall be paid on or before the 7th day of each month, without fail.",
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample_1.pdf",
        "prompt": "Is the security deposit refundable?",
        "expected_answer": (
            "Yes. The Tenant pays an interest-free refundable security deposit, which the Owner "
            "refunds when the Tenant hands over possession of the premises, after adjusting any "
            "dues or damages caused by the Tenant's negligence (normal wear and tear and acts of "
            "god are exempted)."
        ),
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample_1.pdf",
        "prompt": "What is the notice period to terminate this rental agreement?",
        "expected_answer": (
            "The agreement can be terminated before the expiry of the tenancy period by either "
            "party serving one month's prior notice in writing."
        ),
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample_1.pdf",
        "prompt": "Who is responsible for structural or major repairs to the property?",
        "expected_answer": (
            "Day-to-day minor repairs are the Tenant's responsibility at their own expense, but "
            "any structural or major repairs shall be carried out by the Owner."
        ),
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample_1.pdf",
        "prompt": "What happens if the tenant does not vacate the premises when the rent period ends?",
        "expected_answer": (
            "The Tenant will pay damages calculated at two times the rent for the period of "
            "continued occupation after the Rent period expires, and this does not prevent the "
            "Owner from also initiating legal proceedings to recover possession."
        ),
    },
    # -- rental_agreement_sample2.pdf (LESSOR / LESSEE commercial lease) --
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample2.pdf",
        "prompt": "What is the term of this lease?",
        "expected_answer": "The Term of Lease shall be for a fixed period of 11 months, commencing from the effective date.",
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample2.pdf",
        "prompt": "What happens if the lease amount is paid late?",
        "expected_answer": (
            "If the lease amount is paid after the fifth day of the month, late charges apply per "
            "month, and if the Lessee's cheque is dishonoured, the Lessee must pay return cheque "
            "charges to the Lessor."
        ),
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample2.pdf",
        "prompt": "What is the notice period for terminating this lease?",
        "expected_answer": (
            "The agreement may be terminated by either party by issuing a notice thirty days prior "
            "to the agreed expiry of the lease period."
        ),
    },
    {
        "contract_type": "rental",
        "source_file": "sanad/sample_docs/rental/rental_agreement_sample2.pdf",
        "prompt": "Is the security deposit for this lease refundable?",
        "expected_answer": "Yes, the security deposit is repayable at the termination of the agreement, without interest.",
    },
    # -- freelance_agreement_sample2.pdf (Consultancy Agreement Template, India) --
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf",
        "prompt": "Who owns the intellectual property in the deliverables created under this agreement?",
        "expected_answer": (
            "The Deliverables are deemed 'work for hire' and all Intellectual Property Rights in "
            "them vest solely with the Company upon creation."
        ),
    },
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf",
        "prompt": "Is the consultant considered an employee of the company?",
        "expected_answer": (
            "No. The Consultant is engaged as an independent contractor, and nothing in the "
            "agreement is to be construed as creating an employment relationship, agency, "
            "partnership, or joint venture."
        ),
    },
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf",
        "prompt": "Can the consultant work with other clients while under this agreement?",
        "expected_answer": (
            "Yes. The Services are not exclusive to the Company, and the Consultant may enter into "
            "similar arrangements with third parties without the Company's knowledge or consent, "
            "as long as it doesn't breach the Consultant's obligations under the agreement."
        ),
    },
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf",
        "prompt": "What law governs this agreement?",
        "expected_answer": "The agreement is interpreted in accordance with the substantive laws of the Republic of India.",
    },
    # -- service_agreement_sample1.pdf (SERVICE AGREEMENT, Consultant/Company, India) --
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/service_agreement_sample1.pdf",
        "prompt": "On what grounds can the company terminate this agreement immediately?",
        "expected_answer": (
            "The Company may terminate immediately on written notice if the Consultant is unable "
            "to perform or has materially/repeatedly breached the agreement, fails to meet required "
            "qualifications or acts against the Company's reputation or interests, performs "
            "unsatisfactorily (with a 30-day cure period if remediable), or loses required "
            "regulatory/professional body membership."
        ),
    },
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/service_agreement_sample1.pdf",
        "prompt": "Does this agreement create an employer-employee relationship?",
        "expected_answer": (
            "No. The arrangement is on a principal-to-principal basis and does not create any "
            "employee-employer relationship, partnership, or joint venture between the parties."
        ),
    },
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/service_agreement_sample1.pdf",
        "prompt": "How are disputes resolved under this agreement?",
        "expected_answer": (
            "Disputes are first settled amicably between the parties; if not settled amicably, the "
            "dispute is referred to a sole arbitrator appointed by both parties under the "
            "Arbitration & Conciliation Act, 1996, and the arbitrator's award is final and binding."
        ),
    },
    # -- freelance_agreement_sample1.pdf (Client / Freelancer) --
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/freelance_agreement_sample1.pdf",
        "prompt": "Is the freelancer considered an employee of the client company?",
        "expected_answer": (
            "No. The Freelancer agrees that they are an independent contractor, not an employee of "
            "the Client Company/LLP, and is responsible for reporting their own income to the "
            "appropriate authorities."
        ),
    },
    # -- service_agreement_sample.pdf (Service Provider Agreement, India-governed) --
    {
        "contract_type": "freelance",
        "source_file": "sanad/sample_docs/freelance/service_agreement_sample.pdf",
        "prompt": "What is the notice period to terminate this service provider agreement?",
        "expected_answer": (
            "Either Party can terminate the agreement by giving 30 days' prior notice in writing; "
            "however, the Company may also terminate with or without notice at any time if it forms "
            "an opinion that the Agency is providing deficient services."
        ),
    },
]
