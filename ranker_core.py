"""
ranker_core.py - single source of truth for the Redrob hybrid candidate ranker.

Imported by:
  - precompute.py : builds the cached embedding artifacts (slow, offline, GPU/network OK)
  - rank.py       : the <=5-min CPU/no-network ranking step that produces submission.csv

No file IO or heavy work happens at import time - everything is pure functions over
candidate dicts, so this module is safe to import inside the constrained ranking step.

The structured scoring, honeypot battery, behavioral multiplier, and hybrid blend here are
the exact logic developed in explore.ipynb + ranker.ipynb (kept in lockstep with them).
"""
import json
import gzip
import collections
import datetime as dt
import re

import numpy as np
import pandas as pd

# Pinned reference date so reproduction is deterministic regardless of when rank.py runs.
# (Matches the date the notebooks last generated the submission with.)
REFERENCE_DATE = dt.date(2026, 6, 30)
TODAY = REFERENCE_DATE

# Embedding model + sequence length used by precompute.py (cached; not loaded during ranking).
EMB_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LENGTH = 256

# JD distilled to fit-theory, used as the bge retrieval query (needs the query prefix).
JD_QUERY = (
    "Represent this sentence for searching relevant passages: "
    "Senior AI/ML engineer who has built and shipped ranking, retrieval, search, recommendation and "
    "matching systems to real users in production at product companies. Embeddings-based retrieval, "
    "vector search, hybrid search, learning-to-rank, semantic search; rigorous evaluation with NDCG, MRR, "
    "MAP, A/B testing and offline-online correlation. Strong Python. 5-9 years experience, applied ML at "
    "product companies rather than pure services. Based in or willing to relocate to Pune or Noida, India."
)

TOKEN_RE = re.compile("[a-z0-9.+#-]+")  # word tokenizer (no substring false-positives)

# ---- domain constants derived from the JD ----------------------------------------------------
INDIA_CITIES = {'bangalore','bengaluru','pune','noida','hyderabad','mumbai','delhi','gurgaon','gurugram','chennai','kolkata','ncr','new delhi'}
JD_CITIES = {'pune','noida','delhi','gurgaon','gurugram','ncr','new delhi'}
SECONDARY_TIER1_CITIES = {'mumbai','hyderabad','bangalore','bengaluru'}
CONSULTING = {'tcs','tata consultancy','infosys','wipro','accenture','cognizant','capgemini','hcl','tech mahindra','ltimindtree','mindtree','mphasis','dxc', 'larsen toubro','l&t infotech','larsen & toubro','l&t technology services','l&t technology','l&t','larsen and toubro','larsen and toubro infotech','larsen & toubro infotech','larsen & toubro technology services','larsen & toubro technology','capco','ibm consulting','ibm global business services','ibm gbs','hexaware', 'nagarro', 'persistent systems','persistent','ust global','ust','zensar','zensar technologies','zifo'}
INDIA_EMPLOYER_HINTS = CONSULTING | {'flipkart','swiggy','zomato','ola','paytm','razorpay','phonepe','freshworks','zoho','meesho','myntra','zerodha','groww','delhivery','makemytrip','inmobi','jio','reliance','hdfc','icici','axis bank','redrob','sarvam','krutrim'}
SERVICE_INDS = {'it services','consulting','management consulting','outsourcing','staffing','bpo','business process outsourcing','professional services'}
PRODUCT_INDS = {'software','fintech','e-commerce','saas','ai/ml','food delivery','edtech','adtech','transportation','insurance tech'}
RETRIEVAL_EVIDENCE = ['recommendation system','ranking system','ranking model','search relevance','retrieval system','semantic search','vector search','recsys','personalization','learning to rank','information retrieval','embedding','candidate ranking','relevance model','matching system','search ranking','hybrid search','bm25','reranking','retrieval quality']
VECTOR_INFRA = ['pinecone','weaviate','qdrant','milvus','opensearch','elasticsearch','faiss','vector database','vector db','hybrid search','bm25','ann index','hnsw','index refresh','embedding drift','retrieval-quality regression','retrieval quality regression']
PROD_EVIDENCE = ['production','deployed','in prod','real users','at scale','serving','latency','a/b test','ab test','online experiment','index refresh','embedding drift','regression in production','production system']
EVAL_EVIDENCE = ['ndcg','ndcg@','mrr','mrr@','mean reciprocal','map@','mean average precision','precision@','recall@','offline metric','offline-to-online','offline to online','a/b test','ab test','evaluation framework','eval framework','ranking evaluation','relevance evaluation']
PYTHON_EVIDENCE = ['python','pandas','numpy','scipy','sklearn','scikit-learn','pytest','fastapi','pydantic','asyncio','sqlalchemy','pyarrow']
NICE_TO_HAVE = ['lora','qlora','peft','fine-tun','fine tuning','learning to rank','xgboost','lambdamart','hr-tech','recruiting tech','marketplace','distributed systems','large-scale inference','inference optimization','open source','opensource','github contribution','contributed to']
FRAMEWORK_DEMO = ['langchain','llamaindex','prompt engineering','tutorial','demo app','toy project','how i used','blog post']
EXTERNAL_VALIDATION = ['open source','opensource','github','publication','paper','conference','talk','speaker','patent','kaggle','maintainer']
ML_EVIDENCE = ['machine learning','deep learning','neural network','model training','trained a model','classifier','regression model','recommendation','ranking','embedding','natural language',' nlp','computer vision','predictive model','ml model','ml pipeline','feature engineering','xgboost','pytorch','tensorflow','fine-tun','llm','transformer','clustering']
LLM_RECENT = ['langchain','llamaindex','openai api','gpt-4','gpt4','prompt engineering','chatgpt']
RESEARCH_HINT = ['research scientist','phd','postdoc','research fellow','research lab','academic','publication']
NONENG_TITLES = ['marketing','sales','recruiter','human resources','hr manager','product manager','designer','accountant','operations','content writer','customer success','business analyst','project manager']
ML_TITLE = ['ml engineer','machine learning','ai engineer','applied scientist','data scientist','research engineer','nlp engineer','search engineer','ranking engineer','mlops']
CV_SPEECH = ['image classification','object detection','speech recognition','tts','computer vision','robotics','ocr','asr','segmentation','pose estimation']
CV_TITLE_TERMS = ['computer vision','vision engineer','robotics','speech','image','perception']  # CV/speech-primary current-title terms
AI_SKILL_KW = {'rag','retrieval','embedding','embeddings','vector','faiss','pinecone','weaviate','qdrant','milvus','llm','llms','nlp','fine-tuning llms','lora','qlora','peft','ranking','recommendation','transformer','semantic search','information retrieval','bm25'}
IR_ASSESS_TOKENS = ['bm25','embedding','faiss','haystack','fine-tun','elasticsearch','retrieval','semantic','vector','rag','llm','hugging face','sentence transform','information retrieval']

# ---- explicit JD scoring weights / penalties -------------------------------------------------
STRUCT_WEIGHTS = {
    # YOE is a LIGHT positive feature here (~35% of prior weight); the seniority-band
    # preference/guardrail lives mainly in experience_factor() to avoid double-counting.
    'yoe_target': 0.65,
    'yoe_ideal': 0.80,   # small 6-8 peak retained
    'yoe_greater_target': 0.45,
    'yoe_slightly_below_target': 0.40,
    'yoe_below_target': 0.20,
    'yoe_far_below_target': 0.0,
    'applied_ml': 1.0,
    'python_strong': 1.5,
    'python_some': 0.7,
    'ir_assess_strong': 1.5,
    'ir_assess_some': 0.7,
    'retrieval_prod_strong': 1.5,
    'retrieval_prod_some': 0.8,
    'vector_infra_strong': 1.5,
    'vector_infra_some': 0.8,
    'ranking_eval': 1.5,
    # ---- skill-DEPTH bonuses: reward evidence ABOVE the strong/flat thresholds so a
    # candidate with deep retrieval/eval/vector proof outranks one who merely clears the
    # bar. Without these the "strong" tiers saturate (retrieval=5 == retrieval=2), which
    # let ideal-tenure but shallow-skill candidates edge out stronger, slightly-younger ones.
    'retrieval_prod_depth': 0.30,   # per hit of (retrieval+prod) beyond the >=3 "strong" cap (cap +4)
    'retrieval_depth': 0.25,        # per pure-retrieval hit beyond 2 (cap +4)
    'vector_infra_depth': 0.30,     # per vector-infra hit beyond the >=2 "strong" cap (cap +3)
    'ranking_eval_deep': 1.0,       # eval_evidence >= 3 (beyond the flat >=1 credit)
    'ranking_eval_mid': 0.5,        # eval_evidence == 2
    'product_context': 0.4,
    'product_heavy_career': 0.5,
    'service_heavy_with_product': 0.25,
    'profile_complete_high': 0.20,
    'profile_complete_mid': 0.10,
    'profile_complete_low': 0.00,
    'profile_complete_floor': 0.00,
    'assessment_quality_high': 0.30,
    'assessment_quality_mid': 0.20,
    'assessment_quality_low': 0.10,
    'ml_title': 0.3,
    'nice_each': 0.45,
    'nice_cap': 1.6,
}
STRUCT_PENALTIES = {
    'title_chaser': -0.8,
    'framework_demo_only': -0.8,
    'service_only_career': -0.1,
    'consulting_only': -2.0,
    'cv_speech_without_nlp': -1.5,
    'closed_proprietary_no_validation': -0.5,
    'research_without_prod': -0.8,
    'recent_llm_only': -0.8,
    'cv_primary_weak_ir': -1.5,
}
LOCATION_STRUCT_WEIGHTS = {
    'jd_city': 0.6,
    'india_relocate': 0.4,
    'india': 0.1,
    'outside_relocate': 0.0,
    'outside_no_relocate': -0.7,
}

# ---- honeypot / anachronism constants --------------------------------------------------------
TOOL_BIRTH_EXT = {
    'sentence-transformers':2019,'sentence transformers':2019,'sbert':2019,'bge':2023,
    'e5':2022,'instructor embedding':2022,'openai embeddings':2022,'ada-002':2022,'text-embedding':2022,
    'pinecone':2021,'weaviate':2019,'qdrant':2021,'milvus':2019,'opensearch':2021,'faiss':2017,
    'colbert':2020,'splade':2021,'rag':2020,'lora':2021,'qlora':2023,'peft':2022,
    'langchain':2022,'llamaindex':2022,'llama index':2022,
    'chatgpt':2022,'gpt-4':2023,'gpt-3':2020,'gpt-3.5':2022,'llama':2023,'mistral':2023,'mixtral':2023,
    'vllm':2023,'stable diffusion':2022,'whisper':2022,'segment anything':2023,
}
YEARS_RE = re.compile(r'(\d+(?:\.\d+)?)\s*\+?\s*years?')
COMPANY_FOUNDED_AFTER = {
    'sarvam ai': dt.date(2023, 8, 1),
    'krutrim': dt.date(2023, 12, 1),
}
COMPANY_AGE_JD_TERMS = [
    'search','retrieval','ranking','recommendation','recsys','semantic search','vector search',
    'hybrid retrieval','embedding','candidate-jd','ndcg','mrr','offline/online','a/b testing',
    'production ml','learning-to-rank','nlp'
]


# ---- helpers ---------------------------------------------------------------------------------
def lc(s):
    """Lowercase a string, treating None as empty (safe for missing fields)."""
    return (s or '').lower()


def parse_date(s):
    """Parse an ISO date string, returning None on any malformed/missing value."""
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None


def tokens(text):
    """Tokenize text into a set of words (whole-word matching, no substring false-positives)."""
    return set(TOKEN_RE.findall(text))


def tool_present(tool, toks):
    """True if every word of a (possibly multi-word) tool name is present in the token set."""
    return all(pt in toks for pt in tool.split())


def count_hits(text, phrases):
    """Count how many of the given phrases occur as substrings in text."""
    return sum(1 for ph in phrases if ph in text)


def iter_candidates(path):
    """Stream candidate dicts from a .jsonl or .jsonl.gz file."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def work_text(c):
    """Career prose used for the semantic embedding (summary + titles + descriptions; NOT skills)."""
    p = c.get('profile', {}) or {}
    parts = [p.get('summary') or '']
    for j in (c.get('career_history', []) or []):
        parts.append(j.get('title') or '')
        parts.append(j.get('description') or '')
    return '. '.join(t for t in parts if t)


# ---- feature extraction (one candidate) ------------------------------------------------------
def candidate_text_fields(p, ch, cur_title):
    """Return the normalized text blocks reused by feature extractors."""
    summary = lc(p.get('summary'))
    all_desc = ' '.join(lc(j.get('description')) for j in ch)
    all_titles = ' '.join(lc(j.get('title')) for j in ch) + ' ' + cur_title
    company_text = ' '.join(lc(j.get('company')) for j in ch)
    career_text = all_desc + ' ' + all_titles + ' ' + company_text
    text = summary + ' ' + all_desc + ' ' + all_titles
    return text, career_text, company_text


def career_metrics(ch):
    """Aggregate tenure, product/service mix, and applied-ML tenure from career history."""
    durs = [j.get('duration_months') for j in ch if isinstance(j.get('duration_months'), (int, float))]
    avg_stint = (sum(durs) / len(durs)) if durs else None
    short_stints = sum(1 for d in durs if d and d < 20)
    applied_ml_months = 0; service_months = 0; product_months = 0; career_months = 0
    for j in ch:
        t = lc(j.get('title')); jd = lc(j.get('description')); d = j.get('duration_months') or 0
        ind = lc(j.get('industry')); comp = lc(j.get('company'))
        career_months += d
        is_service = ind in SERVICE_INDS or any(cf in comp for cf in CONSULTING)
        is_product = ind in PRODUCT_INDS
        if is_service: service_months += d
        if is_product: product_months += d
        if any(m in t for m in ML_TITLE) or any(ph in jd for ph in ML_EVIDENCE): applied_ml_months += d
    service_share = service_months / career_months if career_months else 0
    product_share = product_months / career_months if career_months else 0
    career_yoe = round(career_months / 12, 2) if career_months else None
    return dict(
        n_jobs=len(ch), avg_stint_months=avg_stint, short_stints=short_stints,
        career_months=career_months, service_months=service_months, product_months=product_months,
        service_share=round(service_share, 3), product_share=round(product_share, 3),
        service_only_career=bool(ch) and service_months == career_months and product_months == 0,
        applied_ml_years=round(applied_ml_months / 12, 2),
        career_yoe=career_yoe,
    )


def grounded_yoe(profile_yoe, career_yoe, ch):
    """Ground claimed YOE against tenure and first work start date."""
    starts = [parse_date(j.get('start_date')) for j in ch if parse_date(j.get('start_date'))]
    span_yoe = round((TODAY - min(starts)).days / 365.25, 2) if starts else None
    yoe_gap_tenure = (profile_yoe - career_yoe) if (profile_yoe is not None and career_yoe is not None) else 0
    yoe_gap_span = (profile_yoe - span_yoe) if (profile_yoe is not None and span_yoe is not None) else 0
    profile_yoe_gt_span = bool(yoe_gap_span > 1.5)
    inflated_yoe = bool(yoe_gap_tenure > 1.5 or profile_yoe_gt_span)
    yoe_candidates = [v for v in [profile_yoe, career_yoe, span_yoe] if v is not None]
    yoe = round(min(yoe_candidates), 2) if inflated_yoe and yoe_candidates else profile_yoe
    return dict(
        yoe=yoe, profile_yoe=profile_yoe, span_yoe=span_yoe,
        yoe_gap_span=round(yoe_gap_span, 2),
        profile_yoe_gt_span=profile_yoe_gt_span,
        inflated_yoe=inflated_yoe,
    )


def assessment_signals(sig):
    """Summarize Redrob skill assessments, including retrieval/IR-specific scores."""
    ass = sig.get('skill_assessment_scores', {}) or {}
    ass_values = list(ass.values())
    ir_scores = [v for k, v in ass.items() if any(tk in k.lower() for tk in IR_ASSESS_TOKENS)]
    ir_assess = max(ir_scores) if ir_scores else 0
    return dict(
        ir_assess=ir_assess,
        assess_avg=(np.mean(ass_values) if ass_values else None),
        assess_max=(max(ass_values) if ass_values else None),
    )


def behavioral_fields(sig):
    """Extract Redrob availability/hireability fields used downstream."""
    la = parse_date(sig.get('last_active_date'))
    days_inactive = (TODAY - la).days if la else None
    return dict(
        response_rate=sig.get('recruiter_response_rate'),
        avg_response_hours=sig.get('avg_response_time_hours'),
        days_inactive=days_inactive,
        open_to_work=sig.get('open_to_work_flag'),
        notice_days=sig.get('notice_period_days'),
        willing_relocate=sig.get('willing_to_relocate'),
        interview_completion=sig.get('interview_completion_rate'),
        offer_acceptance=sig.get('offer_acceptance_rate'),
        github=sig.get('github_activity_score'),
        completeness=sig.get('profile_completeness_score'),
        saved_30d=sig.get('saved_by_recruiters_30d'),
    )


def evidence_counts(text):
    """Phrase-count features derived from profile and career prose, not noisy skills."""
    return dict(
        ml_evidence=count_hits(text, ML_EVIDENCE),
        retrieval_evidence=count_hits(text, RETRIEVAL_EVIDENCE),
        vector_infra_evidence=count_hits(text, VECTOR_INFRA),
        prod_evidence=count_hits(text, PROD_EVIDENCE),
        eval_evidence=count_hits(text, EVAL_EVIDENCE),
        python_evidence=count_hits(text, PYTHON_EVIDENCE),
        nice_to_have_hits=count_hits(text, NICE_TO_HAVE),
        framework_demo_hits=count_hits(text, FRAMEWORK_DEMO),
        external_validation_hits=count_hits(text, EXTERNAL_VALIDATION),
        llm_recent_hits=count_hits(text, LLM_RECENT),
        research_hits=count_hits(text, RESEARCH_HINT),
        cv_speech_hits=count_hits(text, CV_SPEECH),
        nlp_present=('nlp' in text or 'retrieval' in text or 'language model' in text or 'information retrieval' in text),
    )


def extract_row(c):
    """Build the full flat feature row for one candidate: grounded YOE, location/India flags,
    career mix, evidence phrase-counts, assessment + behavioral signals - the input to scoring."""
    p = c.get('profile', {}) or {}
    ch = c.get('career_history', []) or []
    sig = c.get('redrob_signals', {}) or {}
    edu = c.get('education', []) or []
    skills = c.get('skills', []) or []
    cur_title = lc(p.get('current_title'))
    loc = lc(p.get('location')); ctry = lc(p.get('country'))
    text, career_text, company_text = candidate_text_fields(p, ch, cur_title)
    sk_names = [lc(s.get('name')) for s in skills if isinstance(s, dict)]
    career = career_metrics(ch)
    yoe_fields = grounded_yoe(p.get('years_of_experience'), career['career_yoe'], ch)
    assessment = assessment_signals(sig)
    behavior = behavioral_fields(sig)
    evidence = evidence_counts(text)
    prior_india_work_signal = any(ci in career_text for ci in INDIA_CITIES) or 'india' in career_text or any(emp in company_text for emp in INDIA_EMPLOYER_HINTS)
    return dict(
        candidate_id=c['candidate_id'],
        yoe=yoe_fields['yoe'], profile_yoe=yoe_fields['profile_yoe'], career_yoe=career['career_yoe'], span_yoe=yoe_fields['span_yoe'],
        yoe_gap_span=yoe_fields['yoe_gap_span'], profile_yoe_gt_span=yoe_fields['profile_yoe_gt_span'], inflated_yoe=yoe_fields['inflated_yoe'],
        current_title=cur_title,
        current_industry=lc(p.get('current_industry')),
        country=ctry, location=loc,
        india=(ctry == 'india') or any(ci in loc for ci in INDIA_CITIES),
        prior_india_work_signal=prior_india_work_signal,
        jd_city=any(ci in loc for ci in JD_CITIES),
        n_jobs=career['n_jobs'], avg_stint_months=career['avg_stint_months'], short_stints=career['short_stints'],
        career_months=career['career_months'], service_months=career['service_months'], product_months=career['product_months'],
        service_share=career['service_share'], product_share=career['product_share'],
        service_only_career=career['service_only_career'],
        applied_ml_years=career['applied_ml_years'],
        ir_assess=assessment['ir_assess'],
        is_ml_title=any(m in cur_title for m in ML_TITLE),
        is_noneng_title=any(nt in cur_title for nt in NONENG_TITLES),
        consulting_only=bool(ch) and all(any(cf in lc(j.get('company')) for cf in CONSULTING) for j in ch),
        in_product_now=lc(p.get('current_industry')) in PRODUCT_INDS,
        **evidence,
        ai_skill_count=sum(1 for s in sk_names if any(k == s or k in s for k in AI_SKILL_KW)),
        n_skills=len(sk_names),
        edu_tier=(edu[0].get('tier') if edu else None),
        **behavior,
        assess_avg=assessment['assess_avg'],
        assess_max=assessment['assess_max'],
    )


# ---- honeypot impossibility battery ----------------------------------------------------------
def company_age_trap(c, job):
    """Honeypot check: flag a job that starts before its company was founded, but only when
    paired with a suspiciously JD-tailored senior profile (avoids false positives on real gaps)."""
    co = lc(job.get('company'))
    founded = COMPANY_FOUNDED_AFTER.get(co)
    start = parse_date(job.get('start_date'))
    if not founded or not start or start >= founded:
        return None
    p = c.get('profile', {}) or {}
    title_text = lc(p.get('current_title')) + ' ' + lc(p.get('headline')) + ' ' + lc(job.get('title'))
    all_text = ' '.join([
        lc(p.get('headline')), lc(p.get('summary')), lc(job.get('title')), lc(job.get('description')),
        ' '.join(lc(j.get('description')) for j in (c.get('career_history', []) or [])),
    ])
    jd_term_hits = count_hits(all_text, COMPANY_AGE_JD_TERMS)
    senior_search_applied = any(t in title_text for t in ['senior', 'lead', 'staff', 'search engineer', 'applied ml', 'nlp engineer'])
    senior_ds = 'senior data scientist' in title_text
    if (senior_search_applied and jd_term_hits >= 7) or (senior_ds and jd_term_hits >= 6):
        return f'company_before_founding:{co}'
    return None


def exp_claim_anachronism(text):
    """Honeypot check: flag an experience claim ("N years of X") that exceeds the age of the
    named tool X (e.g. more years with a tool than the tool has existed)."""
    for m in YEARS_RE.finditer(text):
        n = float(m.group(1)); ctx = tokens(text[max(0, m.start() - 60):m.end() + 60])
        for tool, birth in TOOL_BIRTH_EXT.items():
            if tool_present(tool, ctx) and n > (2026 - birth) + 1:
                return 'exp_gt_tool_age:' + tool
    return None


def honeypot_reasons(c):
    """Run the full impossibility battery over a candidate and return sorted reason codes
    (two current jobs, future/duration-mismatched dates, tool anachronisms, company-before-
    founding, job/tenure exceeding total YOE, overlapping jobs, expert-skill-with-0-months).
    A non-empty result marks a deliberately planted profile that is dropped before ranking."""
    r = []
    p = c.get('profile', {}) or {}; ch = c.get('career_history', []) or []
    y = p.get('years_of_experience') or 0
    if sum(1 for j in ch if j.get('is_current')) > 1:
        r.append('two_current_jobs')
    for j in ch:
        s = parse_date(j.get('start_date')); e = parse_date(j.get('end_date')); dm = j.get('duration_months')
        if (s and s > TODAY) or (e and e > TODAY):
            r.append('future_date')
        if s and e and isinstance(dm, (int, float)):
            real = (e.year - s.year) * 12 + (e.month - s.month)
            if real < 0 or abs(real - dm) > 10:
                r.append('dur_mismatch')
        if y and isinstance(dm, (int, float)) and dm / 12 > y + 1.5:
            r.append('job_gt_yoe')
        d = lc(j.get('description')); toks = tokens(d)
        if e:
            for tool, birth in TOOL_BIRTH_EXT.items():
                if e.year < birth and tool_present(tool, toks):
                    r.append('anachronism:' + tool); break
        cr = company_age_trap(c, j)
        if cr:
            r.append(cr)
        ec = exp_claim_anachronism(d)
        if ec:
            r.append(ec)
    ec = exp_claim_anachronism(lc(p.get('summary')) + ' ' + lc(p.get('headline')))
    if ec:
        r.append('summary_' + ec)
    dated = sorted([(parse_date(j.get('start_date')), parse_date(j.get('end_date'))) for j in ch
                    if parse_date(j.get('start_date')) and parse_date(j.get('end_date'))], key=lambda x: x[0])
    for i in range(len(dated) - 1):
        s1, e1 = dated[i]; s2, e2 = dated[i + 1]
        if s2 < e1 and (min(e1, e2) - s2).days > 365:
            r.append('job_overlap')
    durs = [j.get('duration_months') or 0 for j in ch]
    if y and sum(durs) / 12 > y + 5:
        r.append('tenure_gt_yoe')
    for sk in (c.get('skills') or []):
        if isinstance(sk, dict) and sk.get('proficiency') == 'expert' and sk.get('duration_months') == 0:
            r.append('expert_skill_0mo'); break
    return sorted(set(r))


# ---- structured scoring ----------------------------------------------------------------------
def early_exceptional_fit(r):
    """True if a sub-4.5yr candidate shows strong enough retrieval+production (+eval/vector/IR)
    evidence to be exempted from the early-career disqualifier."""
    return bool(
        r['retrieval_evidence'] >= 2 and
        r['prod_evidence'] >= 2 and
        (r['eval_evidence'] >= 1 or r['vector_infra_evidence'] >= 2 or r['ir_assess'] >= 50)
    )


def disqualification_flags(r, yoe):
    """Hard anti-fit disqualifiers - if any is True the candidate drops to tier 0: inflated YOE,
    under-min experience, non-exceptional early-career, consulting-only, non-eng title without ML,
    CV/speech without NLP, pure research without production."""
    early_career = 4 <= yoe < 4.5
    flags = dict(
        profile_yoe_gt_span=bool(r.get('profile_yoe_gt_span', False)),
        under_min_yoe=yoe < 4,
        early_career_not_exceptional=early_career and not early_exceptional_fit(r),
        consulting_only=bool(r['consulting_only']),
        noneng_without_ml=bool(r['is_noneng_title'] and not r['is_ml_title']),
        cv_speech_without_nlp=bool(r['cv_speech_hits'] >= 2 and not r['nlp_present']),
        research_without_prod=bool(r['research_hits'] >= 3 and r['prod_evidence'] == 0),
    )
    return flags


def yoe_struct_score(yoe):
    """Experience-band points for the structured score, peaking at the JD's ideal 6-8yr band
    (a LIGHT feature; the main seniority guardrail lives in experience_factor)."""
    if 5 <= yoe <= 9:
        return STRUCT_WEIGHTS['yoe_ideal'] if 6 <= yoe <= 8 else STRUCT_WEIGHTS['yoe_target']
    if 9 < yoe <= 11:
        return STRUCT_WEIGHTS['yoe_greater_target']
    if 4.4 <= yoe < 5:
        return STRUCT_WEIGHTS['yoe_slightly_below_target']
    if 4 <= yoe < 4.4:
        return STRUCT_WEIGHTS['yoe_below_target']
    return STRUCT_WEIGHTS['yoe_far_below_target']


def core_skill_score(r):
    """Threshold points for the core JD skills: Python, verified IR assessment, retrieval+
    production evidence, vector infra, and ranking evaluation (strong vs some vs none tiers)."""
    retrieval_prod_hits = r['retrieval_evidence'] + r['prod_evidence']
    score = 0.0
    score += STRUCT_WEIGHTS['python_strong'] if r['python_evidence'] >= 2 else (STRUCT_WEIGHTS['python_some'] if r['python_evidence'] == 1 else 0.0)
    score += STRUCT_WEIGHTS['ir_assess_strong'] if r['ir_assess'] >= 50 else (STRUCT_WEIGHTS['ir_assess_some'] if r['ir_assess'] > 0 else 0.0)
    score += STRUCT_WEIGHTS['retrieval_prod_strong'] if retrieval_prod_hits >= 3 else (STRUCT_WEIGHTS['retrieval_prod_some'] if retrieval_prod_hits >= 1 else 0.0)
    score += STRUCT_WEIGHTS['vector_infra_strong'] if r['vector_infra_evidence'] >= 2 else (STRUCT_WEIGHTS['vector_infra_some'] if r['vector_infra_evidence'] == 1 else 0.0)
    score += STRUCT_WEIGHTS['ranking_eval'] if r['eval_evidence'] >= 1 else 0.0
    return score


def skill_depth_score(r):
    """Graded bonuses for evidence ABOVE the core-skill strong/flat thresholds, so deeper
    demonstrated retrieval/eval/vector proof keeps earning score (skill depth over raw tenure)."""
    retrieval_prod_hits = r['retrieval_evidence'] + r['prod_evidence']
    score = 0.0
    score += STRUCT_WEIGHTS['retrieval_prod_depth'] * min(max(retrieval_prod_hits - 3, 0), 4)
    score += STRUCT_WEIGHTS['retrieval_depth'] * min(max(r['retrieval_evidence'] - 2, 0), 4)
    score += STRUCT_WEIGHTS['vector_infra_depth'] * min(max(r['vector_infra_evidence'] - 2, 0), 3)
    score += STRUCT_WEIGHTS['ranking_eval_deep'] if r['eval_evidence'] >= 3 else (STRUCT_WEIGHTS['ranking_eval_mid'] if r['eval_evidence'] >= 2 else 0.0)
    return score


def product_context_score(r):
    """Points for product-company context: current employer in a product industry, plus whether
    the career is product-heavy vs service-heavy (the JD prefers product over pure services)."""
    score = STRUCT_WEIGHTS['product_context'] if r['in_product_now'] else 0.0
    if r['product_months'] > r['service_months'] and r['product_months'] > 0:
        score += STRUCT_WEIGHTS['product_heavy_career']
    elif r['service_months'] > r['product_months'] and r['product_months'] > 0:
        score += STRUCT_WEIGHTS['service_heavy_with_product']
    return score


def profile_quality_score(r):
    """Small points for profile completeness (high/mid bands); neutral when the signal is missing."""
    completeness = r['completeness'] if pd.notna(r['completeness']) else None
    if completeness is None:
        return 0.0
    if completeness > 60:
        return STRUCT_WEIGHTS['profile_complete_high']
    if completeness >= 40:
        return STRUCT_WEIGHTS['profile_complete_mid']
    return 0.0


def nice_to_have_score(r):
    """Capped points for nice-to-have skill hits (LoRA, learning-to-rank, distributed systems, ...)."""
    nice_cap_count = int(round(STRUCT_WEIGHTS['nice_cap'] / STRUCT_WEIGHTS['nice_each']))
    return min(r['nice_to_have_hits'], nice_cap_count) * STRUCT_WEIGHTS['nice_each']


def location_struct_score(r):
    """Graded location points inside the structured score (JD-city > India-relocate > India >
    outside-relocate > outside-no-relocate). Complements the location_factor availability multiplier."""
    if r['jd_city']:
        return LOCATION_STRUCT_WEIGHTS['jd_city']
    if r['india'] and r['willing_relocate']:
        return LOCATION_STRUCT_WEIGHTS['india_relocate']
    if r['india']:
        return LOCATION_STRUCT_WEIGHTS['india']
    if r['willing_relocate']:
        return LOCATION_STRUCT_WEIGHTS['outside_relocate']
    return LOCATION_STRUCT_WEIGHTS['outside_no_relocate']


def penalty_flags(r, yoe, disq_flags):
    """Soft anti-fit penalty flags (title-chaser, framework-demo-only, service/consulting-only,
    closed-source-no-validation, recent-LLM-only, CV-primary-with-weak-IR). Each subtracts weight."""
    return dict(
        title_chaser=bool(r['short_stints'] >= 3 and (yoe >= 5 or r['avg_stint_months'] and r['avg_stint_months'] < 24)),
        framework_demo_only=bool(r['framework_demo_hits'] >= 2 and r['retrieval_evidence'] == 0 and r['vector_infra_evidence'] == 0 and r['prod_evidence'] == 0 and r['eval_evidence'] == 0),
        service_only_career=bool(r['service_only_career']),
        consulting_only=disq_flags['consulting_only'],
        cv_speech_without_nlp=disq_flags['cv_speech_without_nlp'],
        closed_proprietary_no_validation=bool((r['applied_ml_years'] >= 5 or yoe >= 7) and r['external_validation_hits'] == 0 and (r['github'] or 0) < 30 and r['eval_evidence'] == 0),
        research_without_prod=disq_flags['research_without_prod'],
        recent_llm_only=bool(r['llm_recent_hits'] >= 1 and r['applied_ml_years'] < 1 and r['ir_assess'] == 0 and r['ml_evidence'] < 2),
        cv_primary_weak_ir=bool(any(k in (r['current_title'] or '') for k in CV_TITLE_TERMS) and r['retrieval_evidence'] < 2 and r['ir_assess'] < 50),
    )


def penalty_score(flags):
    """Sum the (negative) penalty weights for whichever penalty flags fired."""
    score = 0.0
    if flags['title_chaser']: score += STRUCT_PENALTIES['title_chaser']
    if flags['framework_demo_only']: score += STRUCT_PENALTIES['framework_demo_only']
    if flags['service_only_career']: score += STRUCT_PENALTIES['service_only_career']
    if flags['consulting_only']: score += STRUCT_PENALTIES['consulting_only']
    if flags['cv_speech_without_nlp']: score += STRUCT_PENALTIES['cv_speech_without_nlp']
    if flags['closed_proprietary_no_validation']: score += STRUCT_PENALTIES['closed_proprietary_no_validation']
    if flags['research_without_prod']: score += STRUCT_PENALTIES['research_without_prod']
    if flags['recent_llm_only']: score += STRUCT_PENALTIES['recent_llm_only']
    if flags['cv_primary_weak_ir']: score += STRUCT_PENALTIES['cv_primary_weak_ir']
    return score


def positive_struct_score(r, yoe):
    """Sum all positive structured components: experience band, applied-ML years, core skills,
    skill depth, product context, profile quality, ML title, nice-to-haves, and location."""
    score = 0.0
    score += yoe_struct_score(yoe)
    score += min(r['applied_ml_years'], 5) / 5 * STRUCT_WEIGHTS['applied_ml']
    score += core_skill_score(r)
    score += skill_depth_score(r)
    score += product_context_score(r)
    score += profile_quality_score(r)
    score += STRUCT_WEIGHTS['ml_title'] if r['is_ml_title'] else 0.0
    score += nice_to_have_score(r)
    score += location_struct_score(r)
    return score


def _fit(r):
    """Compute the raw structured score (positive components minus penalties) and the hard-
    disqualified flag for one candidate. Shared by struct_score() and tier()."""
    yoe = r['yoe'] or 0
    disq_flags = disqualification_flags(r, yoe)
    disq = any(disq_flags.values())
    s = positive_struct_score(r, yoe)
    s += penalty_score(penalty_flags(r, yoe, disq_flags))
    return s, disq


def tier(r):
    """Bucket a candidate by structured fit: -1 honeypot, 0 disqualified, else 1-5 by score
    thresholds. Only tier 5 (the strongest pool) feeds the hybrid ranking."""
    if r['honeypot']:
        return -1
    s, disq = _fit(r)
    if disq:
        return 0
    if s >= 6.2: return 5
    if s >= 4.8: return 4
    if s >= 3.4: return 3
    if s >= 2.2: return 2
    return 1


def struct_score(r):
    """The structured JD-fit score used by the hybrid blend: the raw fit clamped to >= 0, or
    0.0 for honeypots and hard-disqualified candidates."""
    if r['honeypot']:
        return 0.0
    s, disq = _fit(r)
    return 0.0 if disq else max(s, 0.0)


def behavioral_multiplier(r):
    """Consolidated availability/hireability score over 6 redrob behavioral signals
    (JD directive #3). Used as a multiplier on the structured+semantic fit. Range ~0.10..1.00.
    Missing/not-applicable signals fall back to neutral values (never a blind penalty)."""
    resp = r.get('response_rate')                                          # #7 recruiter_response_rate
    resp = float(resp) if pd.notna(resp) else 0.0
    di = r.get('days_inactive')                                            # #3 last_active_date
    di = float(di) if pd.notna(di) else 220.0
    recency = max(0.0, min(1.0, (270 - di) / (270 - 40)))
    interview = r.get('interview_completion')                              # #19 interview_completion_rate
    interview = float(interview) if pd.notna(interview) else 0.6
    hrs = r.get('avg_response_hours')                                      # #8 avg_response_time_hours
    hrs = float(hrs) if pd.notna(hrs) else 120.0
    speed = max(0.0, min(1.0, (168 - hrs) / (168 - 24)))                   # <=24h -> 1, >=1wk -> 0
    otw = 1.0 if r.get('open_to_work') else 0.0                            # #4 open_to_work_flag
    offer = r.get('offer_acceptance')                                      # #20 offer_acceptance_rate
    offer = float(offer) if (pd.notna(offer) and offer is not None and offer >= 0) else 0.55  # -1/missing -> neutral
    score = (0.30 * resp + 0.18 * recency + 0.18 * interview
             + 0.12 * speed + 0.10 * otw + 0.12 * offer)
    return round(0.10 + 0.90 * score, 3)


def build_feature_frame(candidates):
    """One streaming pass: features + honeypot flags. Returns (df, hp_ids)."""
    rows = []
    hp_ids = {}
    for c in candidates:
        rows.append(extract_row(c))
        rs = honeypot_reasons(c)
        if rs:
            hp_ids[c['candidate_id']] = rs
    df = pd.DataFrame(rows)
    # df-level impossibility: applied-ML tenure exceeds total stated experience
    aml_imposs = df[(df.yoe.notna()) & (df.applied_ml_years > df.yoe.astype(float) + 1.5)]
    for cid in aml_imposs.candidate_id:
        hp_ids.setdefault(cid, []).append('aml_gt_yoe')
    df['honeypot'] = df.candidate_id.isin(hp_ids)
    df['tier'] = df.apply(tier, axis=1)
    df['struct_score'] = df.apply(struct_score, axis=1)
    df['behavioral'] = df.apply(behavioral_multiplier, axis=1)
    return df, hp_ids


# ---- logistics + hybrid blend ----------------------------------------------------------------
def notice_factor(days, exceptional=False):
    """Availability by notice period.
      <=30d  : preferred.
      31-60d : acceptable - notice buyout is standard practice in India, so a normal
               1-2 month notice is not a real blocker.
      >60d   : penalized, UNLESS the candidate is exceptional on skills/struct/sem, in
               which case they're worth the wait or a buyout and the penalty is eased.
    """
    if pd.isna(days):
        return 0.40
    d = float(days)
    if d <= 30: return 1.00          # preferred
    if d <= 60: return 0.72          # acceptable (buyout generally feasible) - clear step below <=30
    if d <= 90: return 0.55 if exceptional else 0.25
    return 0.30 if exceptional else 0.10


def is_exceptional_candidate(r):
    """Worth a long-notice buyout: a LOT better on raw skills, OR extraordinary across
    skills AND structured fit AND semantic fit together."""
    skill_raw = ((r.get('retrieval_evidence') or 0) + (r.get('prod_evidence') or 0)
                 + (r.get('vector_infra_evidence') or 0) + (r.get('eval_evidence') or 0))
    a_lot_better_skills = skill_raw >= 9
    extraordinary_all = ((r.get('struct_norm') or 0) >= 0.92
                         and (r.get('sem_norm') or 0) >= 0.92
                         and skill_raw >= 6)
    return bool(a_lot_better_skills or extraordinary_all)


def location_factor(r):
    """Graded location fit. JD-city (Pune/Noida/Delhi NCR) is preferred; any India-based
    candidate willing to relocate gets strong credit; secondary tier-1 (Mumbai/Hyderabad/
    Bangalore) who won't relocate get partial credit; other non-relocating India candidates
    and non-India candidates (no sponsorship) are down-weighted."""
    if bool(r.get('india')):
        if bool(r.get('jd_city')):
            return 1.00
        loc = r.get('location') or ''
        in_secondary_tier1 = any(city in loc for city in SECONDARY_TIER1_CITIES)
        if bool(r.get('willing_relocate')):
            # A relocation-willing candidate already in a tier-1 hub is a stronger bet than one
            # in a smaller city, so tier-1 relocators get full credit and others a little less.
            # Exception: an exceptional candidate (elite skills/struct/sem) is worth relocating
            # regardless of current city, so they get the same full credit as a tier-1 relocator.
            if in_secondary_tier1 or is_exceptional_candidate(r):
                return 0.80
            return 0.65
        if in_secondary_tier1:
            return 0.12
        return 0.05
    if bool(r.get('prior_india_work_signal')) and bool(r.get('willing_relocate')):
        return 0.10
    return -0.25


def logistics_multiplier(r):
    """Availability multiplier = location_factor x notice_factor (with a notice easing for
    exceptional candidates). Applied on top of fit x behavioral in the base score."""
    return round(location_factor(r) * notice_factor(r.get('notice_days'), is_exceptional_candidate(r)), 3)


def assessment_refine(score):
    """Verified-assessment refinement term (0..0.30) that forms the 15% assessment component of
    the final blend; graded by assessment score band, 0 when missing."""
    if pd.isna(score):
        return 0.0
    s = float(score)
    if s > 70: return 0.30
    if s >= 50: return 0.20
    if s >= 30: return 0.10
    return 0.0


def experience_factor(yoe):
    """Prefer the JD's ideal 6-8 year band, with 5-9 as the broader target."""
    if pd.isna(yoe):
        return 0.0
    y = float(yoe)
    if y < 4.5:
        return 0.58
    if y < 5.0:
        return 0.78
    if y < 6.0:
        return 0.92
    if y <= 8.0:
        return 1.00
    if y <= 9.0:
        return 0.92
    if y <= 10.0:
        return 0.78
    return 0.62


def rank(feat, cand_emb, cand_ids, jd_emb):
    """Hybrid blend over the genuine tier-5 pool. `feat` is indexed by candidate_id.
    Returns the top-100 DataFrame in rank order (index = candidate_id)."""
    sem_vec = cand_emb @ jd_emb
    sem_df = pd.DataFrame({'candidate_id': list(cand_ids), 'sem_sim': sem_vec}).set_index('candidate_id')

    R_all = feat[~feat.honeypot].join(sem_df)
    R_all['struct_norm'] = (R_all['struct_score'] / R_all['struct_score'].max()).clip(0, 1)
    lo, hi = R_all['sem_sim'].quantile(0.01), R_all['sem_sim'].quantile(0.99)
    R_all['sem_norm'] = ((R_all['sem_sim'] - lo) / (hi - lo)).clip(0, 1)

    # Skill-assessment: 40% pass floor + struct-based imputation for MISSING scores (4.5+ pool).
    ASSESS_PASS = 40.0
    STRUCT_WEAK_FLOOR = 6.5
    _mainpool = R_all[(R_all.tier == 5) & (R_all.yoe >= 4.5)]
    ASSESS_MEDIAN = round(float(_mainpool['assess_avg'].median()), 1)
    STRUCT_MEDIAN = round(float(_mainpool['struct_score'].median()), 2)
    _s = R_all['struct_score']; _a = R_all['assess_avg']
    _imputed = np.where(R_all['yoe'] < 4.5, 0.0,
                        np.where(_s >= STRUCT_MEDIAN, ASSESS_MEDIAN,
                                 np.where(_s >= STRUCT_WEAK_FLOOR, ASSESS_PASS, 0.0)))
    R_all['assess_eff'] = np.where(_a.notna(), _a, _imputed)

    # Early-career (4.0-4.5) hard assessment gate.
    EARLY_FULL_ASSESS = 65.0
    EARLY_FLOOR_ASSESS = 50.0
    EARLY_PARTIAL_MULT = 0.8
    assess0 = R_all['assess_avg'].fillna(0.0)
    early_band = (R_all['yoe'] >= 4) & (R_all['yoe'] < 4.5)
    early_exceptional = (
        early_band &
        (R_all['retrieval_evidence'] >= 2) &
        (R_all['prod_evidence'] >= 2) &
        ((R_all['eval_evidence'] >= 1) | (R_all['vector_infra_evidence'] >= 2) | (R_all['ir_assess'] >= 50)) &
        (assess0 >= EARLY_FLOOR_ASSESS)
    )
    yoe_ok = (R_all['yoe'] >= 4.5) | early_exceptional
    pass_floor = (R_all['yoe'] < 4.5) | (R_all['assess_eff'] >= ASSESS_PASS)
    eligible = yoe_ok & pass_floor

    R = R_all[(R_all.tier == 5) & eligible].copy()
    if len(R) < 100:
        R = R_all[(R_all.tier == 5) & yoe_ok].copy()

    R['logistics'] = R.apply(logistics_multiplier, axis=1)
    assess0_R = R['assess_avg'].fillna(0.0)
    early_band_R = (R['yoe'] >= 4) & (R['yoe'] < 4.5)
    R['early_band_mult'] = np.where(early_band_R & (assess0_R < EARLY_FULL_ASSESS), EARLY_PARTIAL_MULT, 1.0)
    R['experience_mult'] = R['yoe'].apply(experience_factor)

    R['fit'] = 0.75 * R['struct_norm'] + 0.25 * R['sem_norm']
    R['base_final'] = R['fit'] * R['behavioral'] * R['logistics'] * R['early_band_mult'] * R['experience_mult']
    R['assessment_refine'] = R['assess_eff'].apply(assessment_refine)

    shortlist = R.sort_values(['base_final', 'candidate_id'], ascending=[False, True]).head(150).copy()
    base_lo, base_hi = shortlist['base_final'].min(), shortlist['base_final'].max()
    shortlist['base_norm_150'] = ((shortlist['base_final'] - base_lo) / (base_hi - base_lo)).clip(0, 1) if base_hi > base_lo else 1.0
    shortlist['final'] = 0.85 * shortlist['base_norm_150'] + 0.15 * shortlist['assessment_refine']

    top = shortlist.sort_values(['final', 'base_final', 'candidate_id'], ascending=[False, False, True]).head(100)
    return top


# ---- reasoning + submission rows -------------------------------------------------------------
def evidence_terms(rec):
    """Extract up to 5 human-readable JD-relevant evidence phrases (retrieval/ranking/vector/
    eval terms) from a candidate's prose, for use in the generated reasoning string."""
    text = ' '.join([
        rec.get('profile', {}).get('summary', ''),
        ' '.join((j.get('title', '') + ' ' + j.get('description', '')) for j in rec.get('career_history', []))
    ]).lower()
    term_map = [
        ('hybrid retrieval', 'hybrid retrieval'), ('semantic search', 'semantic search'),
        ('candidate-jd', 'candidate-JD matching'), ('recommendation system', 'recommendation systems'),
        ('ranking model', 'ranking models'), ('ranking system', 'ranking systems'),
        ('learning-to-rank', 'learning-to-rank'), ('bm25', 'BM25'), ('faiss', 'FAISS'),
        ('qdrant', 'Qdrant'), ('milvus', 'Milvus'), ('weaviate', 'Weaviate'), ('pinecone', 'Pinecone'),
        ('elasticsearch', 'Elasticsearch'), ('vector', 'vector search'), ('embedding', 'embedding-based retrieval'),
        ('ndcg', 'NDCG'), ('mrr', 'MRR'), ('a/b test', 'A/B testing'), ('offline', 'offline evaluation'),
        ('production', 'production deployment'), ('serving', 'serving real users'), ('latency', 'latency work'),
    ]
    out = []
    for needle, label in term_map:
        if needle in text and label not in out:
            out.append(label)
    return out[:5]


def make_reasoning(rec, row):
    """A 1-2 sentence justification. Tone tracks the score (so it can't contradict the rank),
    surfaces honest concerns, and only references facts actually present in the record."""
    p = rec.get('profile', {}) or {}
    title = p.get('current_title') or 'engineer'
    industry = p.get('current_industry') or 'tech'
    yoe = row.get('yoe', p.get('years_of_experience'))
    terms = evidence_terms(rec)
    term_txt = ', '.join(terms[:3]) if terms else 'applied ML and production engineering'
    # Sentence 1 - concrete career evidence. All top-100 are strong vs the 100k pool, so we state
    # facts and let the honest caveats below differentiate rank, rather than labelling fit tiers.
    first = f"{yoe}y as {title} in {industry}, with career evidence of {term_txt}."

    # Sentence 2 - key availability & logistics signals, always stated (recruiter response,
    # notice period, location + relocation). Values vary per candidate, so this stays non-templated.
    loc = p.get('location') or p.get('country') or 'location n/a'
    bits = []
    rr = row.get('response_rate')
    if pd.notna(rr):
        bits.append(f"{round(float(rr) * 100)}% recruiter response")
    nd = row.get('notice_days')
    if pd.notna(nd):
        bits.append(f"{int(float(nd))}-day notice")
    if bool(row.get('jd_city')):
        bits.append(f"based in {loc}")
    elif bool(row.get('india')) and bool(row.get('willing_relocate')):
        bits.append(f"in {loc}, willing to relocate to Pune/Noida")
    elif bool(row.get('india')):
        bits.append(f"in {loc}, not open to relocating")
    else:
        bits.append(f"outside India ({loc}), no visa sponsorship")
    if bool(row.get('open_to_work')):
        bits.append("open to work")
    second = "Signals: " + ", ".join(bits) + "."

    # Honest experience caveat, only where it applies.
    try:
        y = float(yoe)
    except Exception:
        y = None
    if y is not None and y < 5:
        second += f" Note: experience is just under the 5-year target ({yoe}y)."
    elif y is not None and y > 9:
        second += f" Note: experience runs above the 5-9 band ({yoe}y)."
    return first + ' ' + second


def build_outputs(top, recs):
    """Returns (top100_records, submission_rows) for top100.json and submission.csv."""
    order = list(top.index)
    out = []
    submission_rows = []
    for rank_i, cid in enumerate(order, start=1):
        r = top.loc[cid]
        rec = recs.get(cid, {})
        reasoning = make_reasoning(rec, r)
        out.append({
            'rank': rank_i,
            'candidate_id': cid,
            'scores': {
                'final': round(float(r['final']), 4),
                'base_final': round(float(r['base_final']), 4),
                'assessment_refine': round(float(r['assessment_refine']), 3),
                'sem': round(float(r['sem_sim']), 4),
                'struct_score': round(float(r['struct_score']), 3),
                'fit': round(float(r['fit']), 4),
                'behavioral': round(float(r['behavioral']), 3),
                'logistics': round(float(r['logistics']), 3),
                'experience_mult': round(float(r['experience_mult']), 3),
                'tier': int(r['tier']),
                'effective_yoe': round(float(r['yoe']), 2),
                'profile_yoe': (None if pd.isna(r.get('profile_yoe')) else round(float(r.get('profile_yoe')), 2)),
                'career_yoe': (None if pd.isna(r.get('career_yoe')) else round(float(r.get('career_yoe')), 2)),
                'span_yoe': (None if pd.isna(r.get('span_yoe')) else round(float(r.get('span_yoe')), 2)),
            },
            'reasoning': reasoning,
            'profile': rec.get('profile', {}),
            'career_history': rec.get('career_history', []),
            'education': rec.get('education', []),
            'skills': rec.get('skills', []),
            'redrob_signals': rec.get('redrob_signals', {}),
        })
        submission_rows.append({
            'candidate_id': cid,
            'rank': rank_i,
            'score': round(float(r['final']), 6),
            'reasoning': reasoning,
        })
    return out, submission_rows
