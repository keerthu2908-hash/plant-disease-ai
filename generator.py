def generate_answer(results):
    if not results:
        return None

    top_result = results[0]

    matched_text = ", ".join(top_result["matched_symptoms"])
    management_text = " ".join(top_result["management"])

    answer = f"""
Most likely disease: {top_result['disease']}

This appears to be a {top_result['type']} disease affecting {top_result['crop']}.

Why this match:
The symptoms you entered are similar to these known symptoms: {matched_text}.

Cause:
{top_result['cause']}.

Suggested management:
{management_text}
"""

    other_matches = results[1:3]

    return {
        "answer": answer,
        "top_result": top_result,
        "other_matches": other_matches
    }