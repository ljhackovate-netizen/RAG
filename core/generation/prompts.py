"""
All prompt templates.
CRITICAL RULE: Zero domain-specific keywords anywhere.
Prompts work for ANY client in ANY industry.
The LLM dynamically understands content and extracts what is relevant.
"""

# ─── System prompts ──────────────────────────────────────────────────────────

CONTENT_GEN_SYSTEM = """You are a professional content writer working for a marketing agency.
You are given:
1. Brand voice and tone rules extracted from this client's own documents and transcripts
2. Factual context passages from this client's knowledge base (transcripts, pricing, services, proposals)
3. A content generation task

Your job:
- Apply the brand voice rules to match this client's exact tone and personality
- Use ONLY facts, figures, names, and details present in the factual context
- If the context contains specific values (prices, measurements, timelines, names), use them exactly
- Do NOT invent any information not present in either context block
- If context does not contain a specific fact, write around it — never fabricate
"""

QA_SYSTEM = """You are an internal knowledge assistant for a marketing agency team.
Answer questions using ONLY the information in the provided context passages.
If the answer is not in the context, say: "I don't have that information in the knowledge base."
Be specific — include exact figures, names, and details from context when available.
"""

BRAND_VOICE_SYSTEM = """You are a brand strategist creating a Brand Voice Guide document.
Analyze the provided documents and extract a comprehensive, structured brand voice guide.

IMPORTANT: Pay special attention to call transcripts — they reveal how this client naturally 
speaks, their vocabulary, their values, and how they describe their work in their own words.
Extract their ACTUAL spoken language, not generic industry language.

Base EVERYTHING on what is actually present in the documents.
Do not assume industry conventions — derive everything from the actual content.
"""

# ─── User-facing prompt templates ────────────────────────────────────────────

CONTEXTUALIZE_PROMPT = """CLIENT: {client_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRAND VOICE & TONE RULES
(Apply these throughout — match this client's exact personality and communication style)
{brand_voice_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIENT KNOWLEDGE BASE
(Facts to inject — use exact figures, names, materials, timelines, prices from here)
{factual_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Rewrite the following generic content to be specific to {client_name}.
- Apply the brand voice rules above to match their exact tone
- Inject their real details from the knowledge base — exact prices, timelines, names, materials
- If a timeline is in context state it explicitly, never say timeline varies. If a location is in context always name it.
- Never use placeholder language like "a local contractor" or "competitive pricing"
- Never invent facts not present in the context above
Content type: {content_type}

For proposal introductions specifically: reference only ONE 
project example maximum. Do not list multiple unrelated client 
projects. The proposal is for one client about one project — 
keep all references scoped to that single context.
GENERIC CONTENT TO TRANSFORM:
{input_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REWRITTEN CONTENT (client-specific, facts from context, voice from brand guide):"""


GENERATE_FROM_SCRATCH_PROMPT = """CLIENT: {client_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRAND VOICE & TONE RULES
(Write in this client's exact tone and personality throughout)
{brand_voice_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIENT KNOWLEDGE BASE
(Use these facts, figures, names, and details — do not invent anything not here)
{factual_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Write a {content_type} for {client_name}.
Topic / focus: {topic}
Target length: {target_length}

Requirements:
- Write in the tone and voice defined above — not generic marketing language
- Use ONLY information present in the knowledge base above
- Include specific figures, names, materials, and details from context
- If a section requires a fact not in context, use appropriately general language
- Do not invent prices, timelines, names, or specifications

OUTPUT:"""


QA_PROMPT = """CONTEXT FROM {client_name} KNOWLEDGE BASE:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

ANSWER (based only on the context above):"""


BRAND_VOICE_EXTRACTION_PROMPT = """Analyze the following documents from {client_name} and produce a comprehensive Brand Voice Guide.

IMPORTANT: Pay special attention to call transcripts — they reveal how this client 
naturally speaks, their actual vocabulary, their values, and how they describe their 
work in their own words. Extract their ACTUAL language, not generic industry language.

DOCUMENTS:
{documents_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a structured Brand Voice Guide with these sections:

## 1. BUSINESS OVERVIEW
What this business does, where they operate, what makes them unique.
(Based only on what is stated in the documents — no assumptions)

## 2. TARGET AUDIENCE
Who their clients are, what problems they solve, what the client journey looks like.
Specific client types or situations mentioned in transcripts.

## 3. TONE & PERSONALITY
How this business communicates. Formal or casual? Technical or accessible?
Specific words and phrases they use frequently in transcripts.
What language to avoid. How they talk about their own work.

## 4. UNIQUE VALUE PROPOSITIONS
What differentiates them from competitors.
Their process, guarantees, specialties — stated in their own words from transcripts.

## 5. KEY FACTS & FIGURES
Specific numbers, price ranges, timelines, or metrics mentioned in the documents.
Exact pricing if present. Exact service details. Real project examples.

## 6. BRAND LANGUAGE
Specific terminology, product names, brand names, or phrases this business uses.
Words or claims to always include. Words or claims to never use.
Phrases extracted directly from how they speak in transcripts.

## 7. CONTENT GUIDELINES
What topics to highlight. What to avoid claiming.
How to describe their process based on how they describe it themselves.

## 8. SAMPLE PHRASES
10 example phrases or sentences that capture this client's authentic voice.
These must be based on or derived from actual language used in the documents.

Base every section ONLY on what is in the provided documents. Do not assume industry norms."""


CONTENT_TYPE_CONFIGS = {
    "blog_post":      {"label": "Blog Post",            "length": "800-1200 words"},
    "service_page":   {"label": "Service Page",          "length": "400-600 words"},
    "homepage_hero":  {"label": "Homepage Hero Section", "length": "150-250 words"},
    "about_us":       {"label": "About Us Section",      "length": "300-500 words"},
    "faq_section":    {"label": "FAQ Section",           "length": "6-10 Q&A pairs"},
    "location_page":  {"label": "Location / City Page",  "length": "400-600 words"},
    "proposal_intro": {"label": "Proposal Introduction", "length": "200-350 words"},
    "social_post":    {"label": "Social Media Post",     "length": "50-150 words"},
}
