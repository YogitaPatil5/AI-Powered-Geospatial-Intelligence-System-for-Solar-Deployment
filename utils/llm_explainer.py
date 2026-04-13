import os

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("No API key found")
        return None

    try:
        from groq import Groq
        return Groq(api_key=key)
    except:
        print("Install groq: pip install groq")
        return None


def build_prompt(score, rank, features, location_name):
    return f"""
Location: {location_name}

Score: {score}/100
Rank: {rank}

Key data:
GHI: {features.get('ghi')}
Slope: {features.get('slope')}
Cloud: {features.get('cloud_pct')}
NDVI: {features.get('ndvi')}
Grid distance: {features.get('grid_km')}
Road distance: {features.get('road_km')}

Explain:
1. Why this score?
2. 3 strengths
3. 3 concerns
4. Investment advice
"""


def explain_location(score, rank, features, location_name="location"):
    client = get_groq_client()

    if client is None:
        return fallback(score, rank, features, location_name)

    prompt = build_prompt(score, rank, features, location_name)

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return res.choices[0].message.content.strip()
    except:
        return fallback(score, rank, features, location_name)


def fallback(score, rank, f, name):
    return f"""
{name} scored {score}/100 ({rank})

Good:
- GHI: {f.get('ghi')}
- Slope: {f.get('slope')}
- Land: {f.get('land_score')}

Concerns:
- Cloud: {f.get('cloud_pct')}
- Grid: {f.get('grid_km')}
- Road: {f.get('road_km')}

Recommendation:
{"Good site" if score >= 65 else "Needs review"}
"""


def batch_explain(df):
    results = []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}/{len(df)}")
        exp = explain_location(
            score=row.get("suitability_score"),
            rank=row.get("rank"),
            features=row.to_dict(),
            location_name=row.get("name", "location"),
        )
        results.append(exp)

    return results